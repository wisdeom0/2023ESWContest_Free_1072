import csv
import os
import random

Chest_day1 = ["Chest Press", "Bench Press", "Weighted dips",
              "Incline Bench Press (Barbell)", "Cable Fly"]
Chest_day2 = ["Chest Press", "Bench Press",
              "Incline Bench Press (Barbell)", "Bench Press (Dumbbell)", "Cable Fly"]
Chest_day3 = ["Bench Press", "Incline Bench Press (Barbell)", "Weighted dips",
              "Incline Press (Dumbbell)", "Decline Dumbbell Bench Press"]

Back_day1 = ["Lat Pull Down", "One Arm Dummbell Row",
             "Arm Pull Down", "Assist Pull Up", "Barbell Row", "Seated Row"]
Back_day2 = ["Arm Pull Down", "Assist Pull Up", "Barbell Row",
             "One Arm Dummbell Row", "Lat Pull Down", "Seated Row"]
Back_day3 = ["Arm Pull Down", "One Arm Dummbell Row",
             "Lat Pull Down", "Assist Pull Up", "Barbell Row", "Seated Row"]

Shoulder_day1 = ["Side Lateral Raise", "Shoulder Press",
                 "Military Press", "Front Raise", "Seated Military Press (Dumbbell)"]
Shoulder_day2 = ["Side Lateral Raise", "Front Raise",
                 "Seated Shoulder Press (Barbell)", "Front Raise", "Seated Military Press (Dumbbell)"]
Shoulder_day3 = ["Side Lateral Raise", "Shoulder Press",
                 "Military Press", "Front Raise", "Seated Military Press (Dumbbell)"]

Leg_day1 = ["Inner Thigh", "Out Thigh", "Babell Squat",
            "Conventional Deadlift", "Leg Curl", "Leg Press", "Leg Extension"]
Leg_day2 = ["Inner Thigh", "Out Thigh", "Babell Squat",
            "Romanian Deadlift", "Leg Curl", "Dubbell Lift", "Leg Extension"]
Leg_day3 = ["Inner Thigh", "Out Thigh", "Babell Squat",
            "Sumo Deadlift", "Leg Curl", "Leg Press", "Leg Extension"]

workout_routines = {
    'Chest1': Chest_day1, 'Chest2': Chest_day2, 'Chest3': Chest_day3,
    'Back1': Back_day1, 'Back2': Back_day2, 'Back3': Back_day3,
    'Shoulder1': Shoulder_day1, 'Shoulder2': Shoulder_day2, 'Shoulder3': Shoulder_day3,
    'Leg1': Leg_day1, 'Leg2': Leg_day2, 'Leg3': Leg_day3}


def check_and_create_file(filename):
    # 지정한 경로에 파일을 만들기 위해 경로를 미리 설정합니다.
    filename1 = os.path.join(filename + "_routine_num.csv")
    filename2 = os.path.join(filename + "_routine.csv")

    if not os.path.isfile(filename1):
        with open(filename1, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Chest1", "Chest2", "Chest3", "Back1", "Back2", "Back3",
                             "Shoulder1", "Shoulder2", "Shoulder3", "Leg1", "Leg2", "Leg3"])
            writer.writerow([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    if os.path.isfile(filename2):
        os.remove(filename2)

    if not os.path.isfile(filename2):
        with open(filename2, mode='w', newline='') as file:
            writer = csv.writer(file)
    return filename  # 실제로 사용된 파일 경로를 반환합니다.


def get_routine(filename, category):
    filename = check_and_create_file(filename)
    filename1 = os.path.join(filename + "_routine_num.csv")

    with open(filename1, mode='r', newline='') as file:
        reader = csv.reader(file)
        workouts = next(reader)
        counts = list(map(int, next(reader)))

    cat_counts = counts[workouts.index(
        category + '1'):workouts.index(category + '3') + 1]
    min_count = min(cat_counts)
    least_performed_workouts = [workouts[i] for i in range(workouts.index(category + '1'), workouts.index(category + '3') + 1)
                                if counts[i] == min_count]
    selected_workout = random.choice(least_performed_workouts)

    counts[workouts.index(selected_workout)] += 1
    filename2 = os.path.join(filename + "_routine.csv")

    with open(filename1, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(workouts)
        writer.writerow(counts)
    with open(filename2, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(workout_routines[selected_workout])
    return workout_routines[selected_workout]
