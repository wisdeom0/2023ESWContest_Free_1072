import time
import csv
import cv2
import os
import json
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

exe_gui = '랫풀 다운'  # 입력: 운동명

# data
data_keypoint_names_2 = ['Nose', 'Left Shoulder', 'Right Shoulder',
                         'Left Elbow', 'Right Elbow', 'Left Wrist', 'Right Wrist',
                         'Left Hip', 'Right Hip', 'Left Knee', 'Right Knee',
                         'Left Ankle', 'Right Ankle']
mp_keypoint_names = ['NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_ELBOW', 'RIGHT_ELBOW',
                     'LEFT_WRIST', 'RIGHT_WRIST', 'LEFT_HIP', 'RIGHT_HIP',
                     'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE']  ###자료형 통일이 필요할듯... list or dictionary?
data_keypoint_names = {'NOSE': 'Nose', 'LEFT_SHOULDER': 'Left Shoulder', 'RIGHT_SHOULDER': 'Right Shoulder',
                       'LEFT_ELBOW': 'Left Elbow', 'RIGHT_ELBOW': 'Right Elbow',
                       'LEFT_WRIST': 'Left Wrist', 'RIGHT_WRIST': 'Right Wrist',
                       'LEFT_HIP': 'Left Hip', 'RIGHT_HIP': 'Right Hip',  # 50
                       'LEFT_KNEE': 'Left Knee', 'RIGHT_KNEE': 'Right Knee',
                       'LEFT_ANKLE': 'Left Ankle', 'RIGHT_ANKLE': 'Right Ankle'}

data_dir = r"C:\Users\cutiy\OneDrive\사진\프로젝트사진\json"  # json
output_dir = r"C:\Users\cutiy\OneDrive\사진\프로젝트사진\json"  # numpy
os.makedirs(output_dir, exist_ok=True)


# functions
def load_exe(exe_type):
    exercises = {
        '스탠딩 사이드 크런치': (1, 33),
        '스탠딩 니업': (33, 49),
        '버피 테스트': (49, 81),
        '스텝 포워드 다이나믹 런지': (81, 113),
        '스텝 백워드 다이나믹 런지': (113, 145),
        '사이드 런지': (145, 177),
        '크로스 런지': (177, 185),
        '굿모닝': (185, 193),
        '프런트 레이즈': (193, 201),
        '업라이트로우': (201, 209),
        '바벨 스티프 데드리프트': (209, 217),
        '바벨 로우': (217, 249),
        '덤벨 벤트오버 로우': (249, 281),
        '바벨 데드리프트': (281, 313),
        '바벨 스쿼트': (313, 329),
        '바벨 런지': (329, 361),
        '오버 헤드 프레스': (361, 377),
        '사이드 레터럴 레이즈': (377, 409),
        '바벨 컬': (409, 441),
        '덤벨 컬': (441, 473),
        '라잉 레그 레이즈': (473, 489),
        '크런치': (489, 505),
        '바이시클 크런치': (505, 513),
        '시저크로스': (513, 545),
        '힙쓰러스트': (545, 553),
        '플랭크': (553, 561),
        '푸시업': (561, 593),
        '니푸시업': (593, 625),
        'Y - Exercise': (625, 633),
        '덤벨 체스트 플라이': (633, 641),
        '덤벨 인클라인 체스트 플라이': (641, 649),
        '덤벨 풀 오버': (649, 665),
        '라잉 트라이셉스 익스텐션': (665, 697),
        '딥스': (697, 713),
        '풀업': (713, 729),
        '행잉 레그 레이즈': (729, 737),
        '랫풀 다운': (737, 752),  # 원래 753
        '페이스 풀': (753, 761),
        '케이블 크런치': (761, 769),
        '케이블 푸시 다운': (769, 785),
        '로잉머신': (785, 817)
    }

    return exercises.get(exe_type)  # 해당하는 운동이 없는 경우는 넣지 않음


def extract_vectors(data, keypoint_name):
    vectors = []
    for idx in data:
        if keypoint_name in data[idx]:
            coord = data[idx][keypoint_name]
            vector = [coord['x'], coord['y'], coord['z']]
            vectors.append(vector)
    return vectors


def calculate_cosine_similarity(vectors_a, vectors_b):
    sum_cosine_similarity = 0.0
    for i in range(0, 15):
        cosine_similarities = cosine_similarity([vectors_a[i] - vectors_a[i + 1]], [vectors_b[i] - vectors_b[i + 1]])
        sum_cosine_similarity += cosine_similarities[0, 0]
    return sum_cosine_similarity


def pose_correction(exe_gui, csvp, pipe_conn):
    start, finish = load_exe(exe_gui)
    csv_path = os.path.join(csvp + '\jsonoutput.csv')  #####
    for i in range(start, finish+1):
        file_path = os.path.join(data_dir, f'D22-1-{i}-3d.json')
        output_file = os.path.join(output_dir, f'state_{i}.npy')

        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
            data_fra = data['frames']

        result = {}

        for index, value in enumerate(data_fra):  # index == frames, value == pts 이하
            keypoints = value['pts']  # key가 신체부위, data가 좌표인 dictionary == keypoints
            keypoints_dict = {}
            for key, values in keypoints.items():
                keypoints_dict[key] = values  # 신체를 key로 하고 좌표를 data로 하는 dictionary to keypoints_dict
                result[index] = keypoints_dict  #

        keypoint_vectors = {}

        for keypoint_name in data_keypoint_names_2:
            vectors = extract_vectors(result, keypoint_name)
            keypoint_vectors[keypoint_name] = vectors

        np.save(output_file, keypoint_vectors)

        print('Data Setup')

        count = 0
        prev_time = 0
        input_appending = {}
        input_vector = {}
        stage_num = 1  # stages
        stage_check = 0

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:  # 100
                frame = pipe_conn.recv()

                curr_time = time.time()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                result = pose.process(image)  # process done
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                landmarks = result.pose_landmarks.landmark

                input_dict = {}

                for keypoint_name in mp_keypoint_names:
                    x = landmarks[getattr(mp_pose.PoseLandmark, keypoint_name).value].x
                    y = landmarks[getattr(mp_pose.PoseLandmark, keypoint_name).value].y
                    z = landmarks[getattr(mp_pose.PoseLandmark, keypoint_name).value].z

                    input_dict[data_keypoint_names[keypoint_name]] = [x, y, z]

                if curr_time - prev_time > 0.25:
                    # get input_dict, append to input_vector
                    prev_time = curr_time
                    stage_check += 1
                    if stage_check > 4:
                        stage_num += 1
                        print(f'Stage: {stage_num}')
                        stage_check = 0
                        if stage_num > 3:
                            stage_num = 0

                    if len(input_appending) == 0:
                        for keypoint_name in data_keypoint_names_2:
                            input_appending[keypoint_name] = [input_dict[keypoint_name]]  # 130

                    else:
                        for keypoint_name in data_keypoint_names_2:
                            input_appending[keypoint_name].append(input_dict[keypoint_name])
                            count += 1

                    if count > 240:
                        input_vector = input_appending.copy()  # returning 16 frame input_vector
                        count = 0
                        input_appending.clear()

                        # 최대 유사도와 해당 state 파일의 숫자 부분을 저장할 변수 초기화
                        max_similarity = -1
                        most_similar_state_number = None

                        for i in range(start, finish):
                            state_file_path = os.path.join(output_dir, f'state_{i}.npy')
                            state_vectors = np.load(state_file_path, allow_pickle=True).item()

                        similarity_score = 0.0

                        for keypoint_name in data_keypoint_names_2:  # 150
                            input_vector_nparr = np.array(input_vector[keypoint_name])
                            state_vectors_nparr = np.array(state_vectors[keypoint_name])  # key error

                            similarity_score += np.mean(
                                calculate_cosine_similarity(input_vector_nparr, state_vectors_nparr))

                        # 최대 유사도와 해당 state 파일의 숫자 부분 업데이트
                        if similarity_score > max_similarity:
                            max_similarity = similarity_score
                            most_similar_state_number = i

                        if most_similar_state_number is None:  # 가장 유사한걸 못찾으면 이전 단계 사용
                            most_similar_state_number = i

                        # 동작 상태 및 개선 방향 출력
                        file_path_2d = os.path.join(data_dir, f'D22-1-{most_similar_state_number}.json')

                        # JSON 파일 로드
                        with open(file_path_2d, 'r', encoding='utf-8') as json_file:
                            data_state_2d = json.load(json_file)
                            data_type_info = data_state_2d['type_info']

                        # list 작성

                        result_list = []

                        print("동작 정확도:")
                        for condition_dict in data_type_info['conditions']:
                            condition = condition_dict['condition']
                            value = condition_dict['value']

                            value_state = 'O' if value else 'X'

                            print(f"{condition}: {value_state}")
                            result_list.append(condition)
                            result_list.append(value_state)

                        print('\n')

                        print(f"수정할 동작: {data_type_info['description']}")
                        result_list.append(data_type_info['description'])
                        print(result_list)
                        if not os.path.isfile(csv_path) :
                            with open(csv_path, mode='w', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow(result_list)
                        else :
                            with open(csv_path, mode='a', newline='') as file:  # 'a' for append
                                writer = csv.writer(file)
                                writer.writerow(result_list)

                # cv print

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    os.remove(csv_path)
                    break

            cap.release()
            cv2.destroyAllWindows()

    return result_list


###############
# GUI 연결시 삭제
