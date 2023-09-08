from __future__ import print_function
import csv, multiprocessing, cv2, os
import ssl

import numpy as np
import urllib
import urllib.request

from tqdm import tqdm


# COPY from https://github.com/GWUvision/Hotels-50K and fix trouble with ssl

def url_to_image(url):
    # Create a context that doesn't verify SSL certificates
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE

    # Install the custom opener with the SSL context
    urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=context)))

    # Now, you can open URLs without SSL certificate verification
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    return image


# chain,hotel,im_source,im_id,im_url
def download_and_resize(imList):
    for im in imList:
        try:
            saveDir = os.path.join('./images/train/', im[0], im[1], im[2])
            if not os.path.exists(saveDir):
                os.makedirs(saveDir)

            savePath = os.path.join(saveDir, str(im[3]) + '.' + im[4].split('.')[-1])

            if not os.path.isfile(savePath):
                img = url_to_image(im[4])
                if img.shape[1] > img.shape[0]:
                    width = 640
                    height = round((640 * img.shape[0]) / img.shape[1])
                    img = cv2.resize(img, (width, height))
                else:
                    height = 640
                    width = round((640 * img.shape[1]) / img.shape[0])
                    img = cv2.resize(img, (width, height))
                cv2.imwrite(savePath, img)
                print('Good: ' + savePath)
            else:
                print('Already saved: ' + savePath)
        except Exception as e:
            print('Bad: ' + savePath + f" - {e}. {im[4]}")

# chain,hotel,im_source,im_id,im_url
def get_size(img):
    try:
        img = url_to_image(img[4])
        return img.shape
    except Exception as e:
        print(f'Bad: {e}')


def download_images(hotels_ids, dir="../hotels-50k"):
    hotel_f = open(f'./{dir}/dataset/hotel_info.csv', 'r')
    hotel_reader = csv.reader(hotel_f)
    hotel_headers = next(hotel_reader, None)
    hotel_to_chain = {}
    for row in hotel_reader:
        hotel_to_chain[row[0]] = row[2]

    train_f = open(f'./{dir}/dataset/train_set.csv', 'r')
    train_reader = csv.reader(train_f)
    train_headers = next(train_reader, None)

    images = data_to_tuple(hotel_to_chain, train_reader, hotels_ids)

    pool = multiprocessing.Pool()
    NUM_THREADS = multiprocessing.cpu_count()

    imDict = {}
    for cpu in range(NUM_THREADS):
        pool.apply_async(download_and_resize, [images[cpu::NUM_THREADS]])
    pool.close()
    pool.join()


def data_to_tuple(hotel_to_chain, train_reader, hotels_ids):
    images = []
    for im in train_reader:
        im_id = im[0]
        im_url = im[2]
        im_source = im[3]
        hotel = im[1]
        chain = hotel_to_chain[hotel]
        if int(hotel) in hotels_ids:
            images.append((chain, hotel, im_source, im_id, im_url))
    return images

def get_size_images(dir="../hotels-50k"):
    hotel_f = open(f'./{dir}/dataset/hotel_info.csv', 'r')
    hotel_reader = csv.reader(hotel_f)
    hotel_headers = next(hotel_reader, None)
    hotel_to_chain = {}
    for row in hotel_reader:
        hotel_to_chain[row[0]] = row[2]

    train_f = open(f'./{dir}/dataset/train_set.csv', 'r')
    train_reader = csv.reader(train_f)
    images = data_to_tuple(hotel_to_chain, train_reader)
    sizes = []
    for img in tqdm(images):
        sizes.append(get_size(img))
    return sizes

