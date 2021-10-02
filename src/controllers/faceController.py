from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import face_recognition as fr
import numpy as np
import cv2
import os


class FaceController:

    def __init__(self, pathImage):
        self.pathImage = pathImage
        faceLists = self.getEncodeOriginalFace()
        self.faceOriginalEncodes = list(faceLists.values())
        self.faceNames = list(faceLists.keys())
        self.unknownName = "Unknown"
        self.camera()

    def camera(self):

        prototxtPath = r".\services\face_detector\deploy.prototxt"
        weightsPath = r".\services\face_detector\res10_300x300_ssd_iter_140000.caffemodel"
        faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

        maskNet = load_model("./services/mask_detector/mask_detector.model")
        camera = VideoStream(src=0).start()
        while True:

            frame = camera.read()

            (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

            faceLocation = fr.face_locations(frame)
            faceEncoding = fr.face_encodings(frame, faceLocation)
            for (box, pred) in zip(locs, preds):

                (maskTop, maskRight, maskBottom, maskLeft) = box
                (mask, withoutMask) = pred

                label = "Mask" if mask > withoutMask else "No Mask"
                maskColor = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                if mask > withoutMask:
                    # header
                    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                    name = "Covid19"
                    cv2.putText(frame, label, (maskTop, maskRight - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, maskColor, 2)
                    # body
                    cv2.rectangle(frame, (maskTop, maskRight), (maskBottom, maskLeft), maskColor, 2)
                    # footer
                    cv2.putText(frame, name, (maskTop, maskLeft + 30), cv2.FONT_HERSHEY_DUPLEX, 0.80, (255, 255, 255), 2)
                else:
                    for (top, right, bottom, left), newFaceEncodes in zip(faceLocation, faceEncoding):

                        resultCompares = fr.compare_faces(self.faceOriginalEncodes, newFaceEncodes)
                        distances = fr.face_distance(self.faceOriginalEncodes, newFaceEncodes)
                        baseMatchIndex = np.argmin(distances)
                        print(resultCompares)
                        print(distances)
                        if resultCompares[baseMatchIndex]:
                            color = (0, 197, 231)
                            name = self.faceNames[baseMatchIndex]
                        else:
                            color = (0, 0, 255)
                            name = self.unknownName

                        # header
                        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
                        cv2.putText(frame, label, (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, maskColor, 2)
                        # body
                        cv2.rectangle(frame, (left, top - 20), (right, bottom + 20), color, 2)
                        # footer
                        cv2.rectangle(frame, (left, bottom - 15), (right, bottom + 20), color, cv2.FILLED)
                        cv2.putText(frame, name, (left, bottom + 15), cv2.FONT_HERSHEY_DUPLEX, 0.80, (255, 255, 255), 2)

            img = cv2.resize(frame, (950, 700))
            cv2.imshow("Face", img)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    def getEncodeOriginalFace(self) -> dict:
        imgEncodeList = {}
        for pathName, folderName, fileName in os.walk(self.pathImage):
            for name in fileName:
                if name.endswith(".jpg") or name.endswith(".png"):
                    face = fr.load_image_file(f"{self.pathImage}/{name}")
                    encoding = fr.face_encodings(face)[0]
                    key = name.split(".")[0]
                    imgEncodeList[key] = encoding

        return imgEncodeList


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the faces detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our faces mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the faces ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the faces and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = maskNet.predict(faces, batch_size=32)

        # return a 2-tuple of the faces locations and their corresponding
        # locations
        return (locs, preds)

# if __name__ == "__main__":
#     start()
