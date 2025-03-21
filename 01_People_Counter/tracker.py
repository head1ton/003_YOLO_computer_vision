import math


class Tracker:
    def __init__(self):
        # 객체의 중심점을 저장할 딕셔너리
        self.center_points = {}
        # 객체 ID를 위한 카운터
        self.id_count = 0

    def update(self, objects_rect):
        # 객체의 바운딩 박스와 ID를 저장할 리스트
        objects_bbs_ids = []

        # 각 객체의 바운딩 박스를 순회하고.
        for rect in objects_rect:
            x, y, w, h = rect
            # 객체의 중심점 계산. 차후 YOLO에서는 x1, y1, x2, y2 에서 (x1 + x2) // 2 같이 하면 됨
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            same_object_detected = False
            # 기존 객체와의 거리 계산
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                # 기존 객체와의 거리가 35이하면 같은 객체로 간주하고.
                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    # print(self.center_points)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # 새로운 객체인 경우
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # 업데이트된 중심점을 저장할 딕셔너리
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # 새로운 중심점으로 업데이트
        self.center_points = new_center_points.copy()
        return objects_bbs_ids