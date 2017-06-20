from sop_testing_shit import segmentation, blur, contourfindrectangle, contourThatHasCentroid
from os import listdir
from os.path import isfile, join
import numpy
from PIL import Image, ImageDraw
import cv2


def main():
    width = 320
    height = 240


    #luentaan tiedostoja

    folder = "kuvat"

    imagenames = [f for f in listdir(folder) if isfile(join(folder, f))]

    imagenames.sort()

    # Looppaa kuvatiedostot
    for image_file_name in imagenames:
        # Lataa kuva

        print image_file_name
        im = Image.open(folder + "/" + image_file_name)
        im = im.convert('RGB')

        im_numpy = numpy.asarray(im)  # http://stackoverflow.com/questions/384759/pil-and-numpy

        im_numpy = im_numpy[:, :, ::-1]  # PNG RGBA muunto BGR

        image = im_numpy

        # tee NumPy taulukko kameran kuvasta
        #image = frame.array

        # muuta kuva mustavalkoiseksi
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # muuta kuva hsv vareiksi
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # suodatus
        # 0 == ei mitaan
        # 1 == Averaging
        # 2 == Gaussian Blurring
        # 3 == Median Blurring
        # 4 == Bilateral Filtering
        image_gray = blur(3, image_gray);

        # segmentointi, jos parametri on:
        # 0 == ei mitaan
        # 1 == threshold (otsu)
        # 2 == canny (kayta 3 == Median Blurring)
        # 3 == tyhja
        # 4 == tyhja
        # 5 == tyhja
        image_bw = segmentation(2, image_gray);

        # testi kuva
        # cv2.imshow("canny", image_bw)
        # key = cv2.waitKey(1) & 0xFF

        # dilation
        # image_bw = cv2.dilate(image_bw, None)

        # valitse edellisen framen(tai yleensa aloitus piste) centroid kuvan keskipisteesta, b napilla

        areafoundfirsttime = True
        centroidx = width / 2
        centroidy = height / 2
        areafound = True

        # etsi pelialue(=cnt) ja sen keskipiste(=origin)
       # if areafoundfirsttime is False:
            # eka frame
        #    cnt, origin, centroidx, centroidy, areafoundfirsttime, areafound = contourfindrectangle(image, image_bw,
        #                                                                                            centroidx,
        #                                                                                            centroidy,
         #                                                                                           areafoundfirsttime,
        #
        #                                                                                             areafound)
        if areafoundfirsttime is True:
            cnt, origin, centroidx, centroidy, areafound = contourThatHasCentroid(image_bw, centroidx, centroidy,
                                                                                  areafound)  # jos tiedetaan edellinen centroid

        print cnt
        draw = ImageDraw.Draw(im)

        for piste in cnt:
            draw.line((piste[0][0] - 2, piste[0][1], piste[0][0] + 2, piste[0][1]), fill=128)
            draw.line((piste[0][0], piste[0][1] - 2, piste[0][0], piste[0][1] + 2), fill=128)
            im.save("kuvat_result" + "/" + image_file_name)


if __name__ == "__main__":
    main()
