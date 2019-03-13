import numpy as np
import imutils
import cv2
import math


class Stitcher:

    def __init__(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # Opencv 3버전을 사용하는지 확인할 변수
        self.isv3 = imutils.is_cv3()

        self.dir_list = images

        # stitch method로 전달할 ratio, reprojThresh
        self.ratio = ratio
        self.reprojThresh = reprojThresh

        # 사진을 받을 때 맨 처음 사진부터 받게 되면 붙게되는 사진에
        # 큰 왜곡이 생기기 때문에 가운데의 사진을 기준으로 붙임.
        self.center = int(len(images) / 2)
        self.base_img = images[self.center]

        self.final_frame = self.stitch(self.base_img, 1)

        # stitching한 결과를 받아서 'result.jpg'파일로 저장후 보여줌.
        cv2.imwrite("result.jpg", self.final_frame)
        cv2.imshow("Result", self.final_frame)
        cv2.waitKey()

    def stitch(self, imageA, round):

        # 1. 이미지가 없을 때 2. 이미지의 갯수를 초과할 때 ( out of index 일 때 !)
        if (len(self.dir_list) < 1) or (len(self.dir_list) == self.center + round) or (self.center + round) < 0:
            return imageA

        # imageA(base image)의 특징점과 디스크립터를 뽑아냄.
        (kpsA, featuresA) = self.detechAndDescribe(imageA)

        # imageB(base image)의 특징점과 디스크립터를 뽑아냄.
        # 사진 리스트의 가운데 이미지를 base image로 지정하는데, next_idx는 그 전, 후 사진을 가져오기 위한 인덱스
        next_idx = self.center + round
        imageB = self.dir_list[self.center + round]
        (kpsB, featuresB) = self.detechAndDescribe(imageB)

        # 두 이미지 사이의 특징을 매칭시킴
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, self.ratio, self.reprojThresh)

        # 두 이미지 사이의 특징점이 없다면 끝.
        if M is None:
            return None

        # 특징점이 있다면, 다시 튜플의 형태로 풀어버림.
        # H는 histography matrix이다.
        (matches, H, status) = M

        inlinerRatio = float(np.sum(status)) / float(len(status))

        p1 = np.array(kpsA)
        p1 = np.array(kpsB)

        H = H / H[2, 2]
        H_inv = np.linalg.inv(H)

        if inlinerRatio > 0.1:
            (min_x, min_y, max_x, max_y) = self.findDimensions(imageB, H_inv)

            # base 이미지와 비교해서 max_x와 max_y를 구함
            max_x = max(max_x, imageA.shape[1])
            max_y = max(max_y, imageA.shape[0])

            move_h = np.matrix(np.identity(3), np.float32)

            if (min_x < 0):
                move_h[0, 2] += -min_x
                max_x += -min_x

            if (min_y < 0):
                move_h[1, 2] += -min_y
                max_y += -min_y

            mod_inv_h = move_h * H_inv

            img_w = int(math.ceil(max_x))
            img_h = int(math.ceil(max_y))

            # 이전 이미지를 변환시켜 새로운 이미지를 만들어냄.
            base_img_warp = cv2.warpPerspective(imageA, move_h, (img_w, img_h))
            next_img_warp = cv2.warpPerspective(imageB, mod_inv_h, (img_w, img_h))
            enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

            # 테두리 만들기
            (ret, data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY),
                                            0, 255, cv2.THRESH_BINARY)

            enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp,
                                        mask=np.bitwise_not(data_map),
                                        dtype=cv2.CV_8U)

            # 이미지 연산
            final_img = cv2.add(enlarged_base_img, next_img_warp,
                                dtype=cv2.CV_8U)

            final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
            _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            max_area = 0
            best_rect = (0, 0, 0, 0)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)

                deltaHeight = h - y
                deltaWidth = w - x

                area = deltaHeight * deltaWidth

                if (area > max_area and deltaHeight > 0 and deltaWidth > 0):
                    max_area = area
                    best_rect = (x, y, w, h)

            if (max_area > 0):
                if round < 0:
                    return final_img

                # 오른쪽을 다 붙연결했으므로 오른쪽도 연결해줌.
                result = self.stitch(final_img, -round)

                # 재귀 : 다음 이미지를 연결함
                return self.stitch(result, round + 1)

            else:
                return self.stitch(imageA, round + 1)

    def detechAndDescribe(self, image):

        # 단일 채널인 grayscale로 전환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 버전 체크 (__init__참고)를 하여 버전에 맞는 SIFT 알고리즘을 사용함.
        # SIFT 결과로 특징 점들과 디스크립터를 튜플의 형태로 받는다.
        if self.isv3:
            descriptor = cv2.xfeatures2d.SIFT_create()
            (kps, features) = descriptor.detectAndCompute(image, None)

        else:
            detector = cv2.AgastFeatureDetector_create("SIFT")
            kps = detector.detect(gray)

            extractor = cv2.BOWImgDescriptorExtractor("SIFT")
            (kps, features) = extractor.compute(gray, kps)

        # 특징점을 KeyPoint 객체에서 numpy 배열로 전환
        kps = np.float32([kp.pt for kp in kps])

        # 특징점과 특징을 튜플로 반환함 eturn a tuple of keypoints and features
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):

        # k-최근접 이웃 알고리즘(knn)을 사용해서 이미지에서 검출해 낸 특징들을 매칭함.
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
        matches = []

        for m in rawMatches:
            # 거리가 일정한 비율 이내인지 확인하다.
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

            # homography 인자로 넣어줄 매칭점을 계산한다.
        if len(matches) > 4:
            # 두 점을 narray로 변환해줌.
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 두 점의 homography를 계산한다.
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # 매칭점과 homorgaphy 행렬, 그리고 모든 매칭된 점의 상태를
            # 튜플의 형태로 같이 넘겨줌.
            return (matches, H, status)

        # 매칭점없으면 None을 반환.
        return None

    def findDimensions(self, image, homography):
        base_p1 = np.ones(3, np.float32)
        base_p2 = np.ones(3, np.float32)
        base_p3 = np.ones(3, np.float32)
        base_p4 = np.ones(3, np.float32)

        (y, x) = image.shape[:2]

        base_p1[:2] = [0, 0]
        base_p2[:2] = [x, 0]
        base_p3[:2] = [0, y]
        base_p4[:2] = [x, y]

        max_x = None
        max_y = None
        min_x = None
        min_y = None

        for pt in [base_p1, base_p2, base_p3, base_p4]:

            hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
            hp_arr = np.array(hp, np.float32)

            normal_pt = np.array([hp_arr[0] / hp_arr[2], hp_arr[1] / hp_arr[2]], np.float32)

            if (max_x == None or normal_pt[0, 0] > max_x):
                max_x = normal_pt[0, 0]

            if (max_y == None or normal_pt[1, 0] > max_y):
                max_y = normal_pt[1, 0]

            if (min_x == None or normal_pt[0, 0] < min_x):
                min_x = normal_pt[0, 0]

            if (min_y == None or normal_pt[1, 0] < min_y):
                min_y = normal_pt[1, 0]

        min_x = min(0, min_x)
        min_y = min(0, min_y)

        return (min_x, min_y, max_x, max_y)
