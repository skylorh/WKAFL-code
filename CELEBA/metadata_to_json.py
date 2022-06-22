import json
import numpy as np
import os

TARGET_NAME = 'Smiling'
parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))


def get_metadata():
    f_identities = open(os.path.join(
        parent_path, 'data', 'raw', 'identity_CelebA.txt'), 'r')
    identities = f_identities.read().split('\n')

    f_attributes = open(os.path.join(
        parent_path, 'data', 'raw', 'list_attr_celeba.txt'), 'r')
    attributes = f_attributes.read().split('\n')

    return identities, attributes


def get_celebrities_and_images(identities):
    all_celebs = {}

    for line in identities:
        info = line.split()
        if len(info) < 2:
            continue
        image, celeb = info[0], info[1]
        if celeb not in all_celebs:
            all_celebs[celeb] = []
        all_celebs[celeb].append(image)

    good_celebs = {c: all_celebs[c] for c in all_celebs if len(all_celebs[c]) >= 5}
    return good_celebs


def _get_celebrities_by_image(identities):
    good_images = {}
    for c in identities:
        images = identities[c]
        for img in images:
            good_images[img] = c
    return good_images


def get_celebrities_and_target(celebrities, attributes, attribute_name=TARGET_NAME):
    col_names = attributes[1]
    col_idx = col_names.split().index(attribute_name)

    celeb_attributes = {}
    good_images = _get_celebrities_by_image(celebrities)

    for line in attributes[2:]:
        info = line.split()
        if len(info) == 0:
            continue

        image = info[0]
        if image not in good_images:
            continue

        celeb = good_images[image]
        att = (int(info[1:][col_idx]) + 1) / 2

        if celeb not in celeb_attributes:
            celeb_attributes[celeb] = []

        celeb_attributes[celeb].append(att)

    return celeb_attributes


def build_json_format(celebrities, targets):
    all_data = {}

    # ['1234', '4567']
    celeb_keys = [c for c in celebrities]

    # [4, 6]
    num_samples = [len(celebrities[c]) for c in celeb_keys]

    # {'x': [xxx.jpg, xxx.jpg], 'y': [1,0,1,0]}
    data = {c: {'x': celebrities[c], 'y': targets[c]} for c in celebrities}

    all_data['users'] = celeb_keys
    all_data['num_samples'] = num_samples
    all_data['user_data'] = data
    return all_data


def write_json(json_data, file_name):
    dir_path = os.path.join(parent_path, 'data', 'CELEBA', 'raw_data')

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    file_path = os.path.join(dir_path, file_name)

    print('writing {}'.format(file_name))
    with open(file_path, 'w') as outfile:
        json.dump(json_data, outfile)


def dict_slice(adict, start, end):
    keys = list(adict.keys())
    dict_slice = {}
    for k in keys[start:end]:
        dict_slice[k] = adict[k]
    return dict_slice


def main():
    identities, attributes = get_metadata()

    train_size = 0.75

    # celebrities {'1234':[xxx.jpg, xxx.jpg], '4567':[xxx.jpg, xxx.jpg]}
    celebrities = get_celebrities_and_images(identities)

    # targets {'1234': [0,1,1,0], '4567':[1,0,1,0]}
    targets = get_celebrities_and_target(celebrities, attributes)

    # targets_train = dict_slice(targets, 0, int(len(targets)*train_size))
    # celebrities_train = dict_slice(celebrities, 0, int(len(celebrities)*train_size))
    targets_train = dict_slice(targets, 0, 6000)
    celebrities_train = dict_slice(celebrities, 0, 6000)
    print(len(targets_train))

    # targets_test = dict_slice(targets, int(len(targets)*train_size), len(targets))
    # celebrities_test = dict_slice(celebrities, int(len(celebrities)*train_size), len(celebrities))
    targets_test = dict_slice(targets, 6000, 9000)
    celebrities_test = dict_slice(celebrities, 6000, 9000)
    print(len(targets_test))


    json_data_train = build_json_format(celebrities_train, targets_train)
    json_data_test = build_json_format(celebrities_test, targets_test)

    write_json(json_data_train, 'train_data.json')
    write_json(json_data_test, 'test_data.json')


if __name__ == '__main__':
    main()