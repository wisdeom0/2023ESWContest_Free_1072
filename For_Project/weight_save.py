import os
import csv
import datetime
import pytz

RM_ESTIMATORS = {
    "Deadlift": {
        '1rm_from_10rm': [21.651, 1.124], '5rm_from_10rm': [20.456, 0.9312], '1rm_from_5rm': [1.195, 1.207]},
    "Squat": {
        '1rm_from_10rm': [29.214, 1.013],'5rm_from_10rm': [18.282, 0.9675], '1rm_from_5rm': [10.922, 1.074]},
    "Bench Press": {
        '1rm_from_10rm': [19.017, 1.047],'5rm_from_10rm': [12.002, 0.9721], '1rm_from_5rm': [7.015, 1.077]},
    "Chest Press": {
        '1rm_from_10rm': [17.170, 1.165],'5rm_from_10rm': [8.626, 1.0777], '1rm_from_5rm': [8.544, 1.165]},
    "Lat Pull Down": {
        '1rm_from_10rm': [9.930, 1.254],'5rm_from_10rm': [-7.24, 1.0829], '1rm_from_5rm': [3.640, 1.254]},
    "Leg Press": {
        '1rm_from_10rm': [34.896, 1.096], '5rm_from_10rm': [15.769, 1.0438], '1rm_from_5rm': [19.127, 1.096]},
    "Leg Curl": {
        '1rm_from_10rm': [34.110, 0.783], '5rm_from_10rm': [11.209, 0.9233], '1rm_from_5rm': [22.901, 0.848]},
}

def estimate_rm(exercise, from_rm, to_rm, rm_value):
    if exercise not in RM_ESTIMATORS:
        return rm_value

    if to_rm == from_rm:
        return rm_value

    if (from_rm, to_rm) in [('10rm', '1rm'), ('1rm', '10rm')]:
        estimator = RM_ESTIMATORS[exercise].get('1rm_from_10rm')
    elif (from_rm, to_rm) in [('10rm', '5rm'), ('5rm', '10rm')]:
        estimator = RM_ESTIMATORS[exercise].get('5rm_from_10rm')
    elif (from_rm, to_rm) in [('1rm', '5rm'), ('5rm', '1rm')]:
        estimator = RM_ESTIMATORS[exercise].get('1rm_from_5rm')
    else:
        return "Unknown RM conversion"

    if from_rm < to_rm:
        return round(estimator[0] + estimator[1] * rm_value)
    else:
        return round((rm_value - estimator[0]) / estimator[1])

def estimate_other_rms(exercise, input_rm, input_weight):
    if input_rm <= 7:
        return {
            "1RM": round(input_weight),
            "5RM": estimate_rm(exercise, '1rm', '5rm', input_weight),
            "10RM": estimate_rm(exercise, '1rm', '10rm', input_weight),
        }
    elif input_rm <= 10:
        rm_1 = estimate_rm(exercise, '5rm', '1rm', input_weight)
        return {
            "1RM": round(rm_1),
            "5RM": round(input_weight),
            "10RM": estimate_rm(exercise, '1rm', '10rm', rm_1),
        }
    elif input_rm <= 15:
        rm_1 = estimate_rm(exercise, '10rm', '1rm', input_weight)
        return {
            "1RM": round(rm_1),
            "5RM": estimate_rm(exercise, '1rm', '5rm', rm_1),
            "10RM": round(input_weight),
        }
    else:
        return {"1RM" : input_weight, "5RM" : input_weight, "10RM" : input_weight}


seoul_tz = pytz.timezone('Asia/Seoul')


def initialize_csv(filename):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Date", "Exercise", "Weight", "Repetitions", "Recommended 1RM", "Recommended 10RM", "Recommended 15RM"])


def last_row(filename, target_exercise):
    if not os.path.exists(filename):
        initialize_csv(filename)
        return None

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        for row in reversed(data):
            if row[1] == target_exercise:
                return row
    return None


def input_and_save(name, exercise, weight, reps):
    filename = f"{name}.csv"

    data = estimate_other_rms(exercise, reps, weight)
    rm_data = list(data.values())

    now = datetime.datetime.now(seoul_tz).strftime("%Y-%m-%d %H:%M")

    last_info = last_row(filename, exercise)
    if last_info:
        rm_data = [max(int(rm_data[0]), int(last_info[-3])),
                   max(int(rm_data[1]), int(last_info[-2])),
                   max(int(rm_data[2]), int(last_info[-1]))]

    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([now, exercise, weight, reps, rm_data[0], rm_data[1], rm_data[2]])
