"""MANSION 시뮬 기반 벤치마크 환경 — 로봇 스폰·텔레포트 제어·엘리베이터 FSM.

설계:
  · 스폰 = 시작 포즈에 로봇 상태(포즈+footprint) 생성, 물리 제어기 없음.
    액션 = 텔레포트 스텝 (dx, dz, dyaw). 스텝마다 footprint 충돌 검사
    (씬 GT 기하 침식 그리드 — 데이터셋과 동일 규약) 실패 시 제자리+충돌 카운트.
  · 관측 = 해당 로봇 sensor_rgb 실높이 RGB-D (ThorDriver 렌더).
  · 엘리베이터 FSM:
      문 닫힘(기본) → [호출: 존 1.2m + 문 방향 ±60° + 연속 2스텝]
      → [대기 w스텝(시드 결정적), 존 이탈 시 리셋] → 문 열림(개폐 렌더)
      → [10스텝 내 미진입 시 닫힘] → [카 내부 footprint 적합 + 체류 T스텝]
      → 목표층 씬 교체 + 도착층 문 앞 배치(도착 시 즉시 열림).
    대기·탑승 스텝은 TL·시간에 가산. 층 버튼은 v1 미평가(에피소드 목표층).
  · 계단: 계단 존 진입 + stairs 능력(GT) 로봇만 층 전환(씬 교체) —
    능력 없는 로봇은 진입해도 전환 불발(에이전트 오판이 실패로 드러남).

Gym류 API:
  env = MansionSimEnv(tag, gpu)
  obs = env.reset(robot_id, start=(floor,x,z,yaw), goal=(floor,x,z),
                  wait_steps=w)
  obs, info = env.step((dx, dz, dyaw))      # info: state/collided/모드 이벤트
  env.trajectory  → [(floor,x,z), ...] (지표 계산용)
"""
import json
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "models"))
import mansion_adapter as ma

CALL_ZONE_M = 1.2
CALL_FACE_DEG = 60.0
CALL_DWELL = 2
DOOR_OPEN_TIMEOUT = 10
RIDE_DWELL = 3


class MansionSimEnv:
    def __init__(self, tag, gpu=0, render=True, tol_cells=3):
        """tol_cells: 충돌 판정 허용 팽창(셀) — expert 궤적 스무딩이 침식
        경계를 스치는 것을 GT 규약대로 허용(00_validate_sim으로 보정,
        expert 리플레이 통과율 ≥99% 게이트)."""
        self.tag = tag
        self.mc = ma._mc()
        self.tol_cells = tol_cells
        self.floors = ma.load_floors(tag)
        self.driver = ma.ThorDriver(tag, gpu=gpu) if render else None
        self._find_transit_zones()

    # ---- 전이 존 추출: common.zone_anchor — **로봇 폭별** 재계산 ----
    def _find_transit_zones(self, w_eff=0.3, h=1.0):
        # floor → (door x,z, normal nx,nz, car cx,cz)
        self.elev = {}
        # floor → (zone x,z)
        self.stairs = {}
        for fl, geom in self.floors.items():
            for kind in ("elev", "stair"):
                a = geom.zone_anchor(kind, w_eff, h)
                if a is None:
                    continue
                (iz, ix), mid = a
                # 문 앞 존(주행 가능 셀)
                zx, zz = geom.to_world(iz, ix)
                if kind == "stair":
                    self.stairs[fl] = (zx, zz)
                    continue
                if mid is None:
                    mid = (zx, zz)
                dx, dz = zx - mid[0], zz - mid[1]
                L = math.hypot(dx, dz) or 1.0
                # 법선 = 문→존 방향
                nx, nz = dx / L, dz / L
                m = geom.elev_mask
                zzs, xxs = np.where(m)
                cw = [geom.to_world(int(r), int(c))
                      for r, c in zip(zzs[::max(1, len(zzs) // 64)],
                                      xxs[::max(1, len(xxs) // 64)])]
                cx = float(np.mean([p[0] for p in cw]))
                cz = float(np.mean([p[1] for p in cw]))
                self.elev[fl] = (mid[0], mid[1], nx, nz, cx, cz)

    # ---- 에피소드 ----
    def reset(self, robot, start, goal, wait_steps=3):
        """robot = {'w_eff','cam_h','stairs_ok'(GT 능력)}."""
        self.robot = robot
        self._pass_cache = getattr(self, "_pass_cache", {})
        # 존 anchor는 로봇 폭 기준으로 재계산 — 광폭 로봇도 존 도달이 가능해야 한다
        zkey = round(robot["w_eff"], 3)
        if getattr(self, "_zone_w", None) != zkey:
            self._find_transit_zones(robot["w_eff"], robot["h"])
            self._zone_w = zkey
        self.fl, self.x, self.z, self.yaw = start
        self.goal = goal
        self.wait_steps = int(wait_steps)
        self.trajectory = [(self.fl, self.x, self.z)]
        self.steps = 0
        self.collisions = 0
        self.state = "driving"
        self.events = []
        # 엘베 FSM
        # (floor, ei) → 남은 열림 스텝
        self._door_open = {}
        # (ei, 충족 스텝 수)
        self._call = None
        self._wait_left = None
        self._ride_dwell = 0
        return self._obs()

    def _free(self, fl, x, z, allow_elev=False):
        """footprint 충돌 — 침식 그리드(passable). 엘베 셀은 주행 그리드
        밖이므로 문 열림 중에만 진입 허용 + 카 폭 적재 검사."""
        key = (fl, round(self.robot["w_eff"], 3))
        if key not in self._pass_cache:
            from scipy import ndimage
            base = self.floors[fl].passable(self.robot["w_eff"],
                                            self.robot["h"])
            if self.tol_cells:
                base = ndimage.binary_dilation(
                    base, iterations=self.tol_cells) \
                    & self.floors[fl].inside
            self._pass_cache[key] = base
        ok = self._pass_cache[key]
        g = self.floors[fl]
        iz, ix = g.to_idx(x, z)
        if not (0 <= iz < ok.shape[0] and 0 <= ix < ok.shape[1]):
            return False
        if ok[iz, ix]:
            return True
        if allow_elev and g.elev_mask[iz, ix]:
            return self._car_fits(fl)
        return False

    def _car_fits(self, fl=None):
        """적재 검사 — 카 내부를 로봇 반경으로 침식해 잔여 셀 존재 여부."""
        fl = self.fl if fl is None else fl
        g = self.floors[fl]
        ck = ("car", fl, round(self.robot["w_eff"], 3))
        if ck not in self._pass_cache:
            from scipy import ndimage
            r = max(1, int(self.robot["w_eff"] / 2 / 0.025))
            self._pass_cache[ck] = ndimage.binary_erosion(
                g.elev_mask, iterations=r)
        return bool(self._pass_cache[ck].any())

    def _obs(self):
        if self.driver is None:
            return None
        rgb, dep = self.driver.render(self.fl, self.x, self.z, self.yaw,
                                      self.robot["cam_h"])
        return {"rgb": rgb, "depth": dep,
                "pose": (self.fl, self.x, self.z, self.yaw)}

    # ---- 엘베 FSM 갱신 (스텝마다) ----
    def _near_elev(self):
        e = self.elev.get(self.fl)
        if e is None:
            return None, False, None
        ex, ez, nx, nz, cx, cz = e
        # 존 중심 = 문 전방
        zx, zz = ex + nx * 0.8, ez + nz * 0.8
        if math.hypot(self.x - zx, self.z - zz) <= CALL_ZONE_M:
            bearing = math.degrees(math.atan2(ex - self.x, ez - self.z))
            dyaw = (bearing - self.yaw + 180) % 360 - 180
            return 0, abs(dyaw) <= CALL_FACE_DEG, (ex, ez)
        return None, False, None

    def _inside_car(self):
        """카 내부 = elev_mask 셀(footprint 중심) — 적재 검사 겸용."""
        if self.fl not in self.elev:
            return None
        g = self.floors[self.fl]
        iz, ix = g.to_idx(self.x, self.z)
        if (0 <= iz < g.elev_mask.shape[0]
                and 0 <= ix < g.elev_mask.shape[1]
                and g.elev_mask[iz, ix]):
            return 0
        return None

    def _update_elevator(self):
        ei, facing, door = self._near_elev()
        # 문 열림 타이머
        for k in list(self._door_open):
            self._door_open[k] -= 1
            if self._door_open[k] <= 0:
                del self._door_open[k]
                self.events.append(("door_close", self.steps))
        # 호출 → 대기 → 열림
        if self._wait_left is not None:
            if ei is None:
                # 존 이탈 → 리셋
                self._wait_left = None
                self.events.append(("call_reset", self.steps))
            else:
                self._wait_left -= 1
                self.state = "waiting"
                if self._wait_left <= 0:
                    self._door_open[(self.fl, ei)] = DOOR_OPEN_TIMEOUT
                    self._wait_left = None
                    self.state = "driving"
                    self.events.append(("door_open", self.steps))
        elif ei is not None and facing:
            self._call = ((ei, self._call[1] + 1)
                          if self._call and self._call[0] == ei
                          else (ei, 1))
            if self._call[1] >= CALL_DWELL and \
                    (self.fl, ei) not in self._door_open:
                self._wait_left = self.wait_steps
                self.events.append(("call", self.steps))
        else:
            self._call = None
        # 탑승 판정
        ci = self._inside_car()
        if ci is not None and (self.fl, ci) in self._door_open:
            self._ride_dwell += 1
            self.state = "in_elevator"
            if self._ride_dwell >= RIDE_DWELL:
                self._transit_elev(ci)
        else:
            self._ride_dwell = 0

    def _transit_elev(self, ci):
        tgt = self.goal[0]
        if tgt not in self.elev:
            return
        ex, ez, nx, nz, cx, cz = self.elev[tgt]
        self.fl = tgt
        # 도착층 문 앞
        self.x, self.z = ex + nx * 1.0, ez + nz * 1.0
        self.yaw = math.degrees(math.atan2(nx, nz))
        # 도착 즉시 열림
        self._door_open = {(tgt, 0): DOOR_OPEN_TIMEOUT}
        self._ride_dwell = 0
        self.state = "driving"
        self.events.append(("elevator_transit", self.steps, tgt))
        self.trajectory.append((self.fl, self.x, self.z))

    def _update_stairs(self):
        if self.fl in self.stairs:
            sx, sz = self.stairs[self.fl]
            if math.hypot(self.x - sx, self.z - sz) <= 1.0:
                if not self.robot["stairs_ok"]:
                    self.events.append(("stairs_denied", self.steps))
                    return
                tgt = self.goal[0]
                if tgt == self.fl:
                    return
                if tgt not in self.stairs:
                    return
                self.fl = tgt
                self.x, self.z = self.stairs[tgt]
                self.events.append(("stairs_transit", self.steps, tgt))
                self.trajectory.append((self.fl, self.x, self.z))
                return

    # ---- 스텝 ----
    def step(self, action, observe=True):
        """observe=False면 관측을 렌더하지 않는다(obs 자리에 None).

        홉 루프는 홉 경계에서만 관측을 쓰므로 홉 내부 저수준 이동에서
        렌더를 돌리면 그만큼이 통째로 낭비다(스텝당 ~350ms).
        """
        dx, dz, dyaw = action
        # dyaw = "이 이동의 도착 heading"이지 독립 회전 행동이 아니다.
        # 제자리 회전(dx=dz=0, dyaw≠0)은 정책 행동 공간에 없으므로 호출
        # 자체를 막는다 — 대기는 완전 정지(0,0,0)로 표현한다.
        assert not (dx == 0.0 and dz == 0.0 and dyaw != 0.0), \
            "순수 회전 금지 — 방향 전환은 이동의 도착 heading으로만"
        self.steps += 1
        # 회전 선적용: 이동 방향·시선 판정이 회전
        # 반영 후 기준 — 구현은 회전 후적용이라 조향이 1스텝 지연돼 코너
        # 스침·충돌 과대 계상이 발생했음(원인 분해로 실증).
        self.yaw = (self.yaw + dyaw) % 360
        rad = math.radians(self.yaw)
        nx = self.x + dx * math.cos(rad) + dz * math.sin(rad)
        nz = self.z - dx * math.sin(rad) + dz * math.cos(rad)
        allow = any(k[0] == self.fl for k in self._door_open)
        # 탑승 = 스냅 이벤트(순간이동 제어 규약): 문 열림 + 존 내 +
        # 전진 액션 → 적재 검사 통과 시 카 중심으로 탑승
        ei, facing, _ = self._near_elev()
        # 재탑승 루프 방지: 목표층 도착 후엔 탑승 불가 + 문 방향 시선 요구
        if (allow and ei is not None and facing and dz > 0
                and self.goal[0] != self.fl and self._car_fits()):
            ex, ez, nx_, nz_, cx, cz = self.elev[self.fl]
            self.x, self.z = cx, cz
            self.trajectory.append((self.fl, self.x, self.z))
            self.events.append(("board", self.steps))
            self._update_elevator()
            self._update_stairs()
            return (self._obs() if observe else None), {"state": self.state,
                                 "collisions": self.collisions,
                                 "steps": self.steps,
                                 "events": self.events[-3:],
                                 "pose": (self.fl, self.x, self.z,
                                          self.yaw)}
        if self._free(self.fl, nx, nz, allow_elev=allow):
            self.x, self.z = nx, nz
        else:
            self.collisions += 1
        self.trajectory.append((self.fl, self.x, self.z))
        self._update_elevator()
        self._update_stairs()
        info = {"state": self.state, "collisions": self.collisions,
                "steps": self.steps, "events": self.events[-3:],
                "pose": (self.fl, self.x, self.z, self.yaw)}
        return (self._obs() if observe else None), info

    def close(self):
        if self.driver is not None:
            self.driver.close()
