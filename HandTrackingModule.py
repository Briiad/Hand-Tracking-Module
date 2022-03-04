import cv2
import mediapipe as mp
import time


class handDetector():
  # Have to use 5 parameters instead of 4
  def __init__(self, mode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
    self.mode = mode
    self.maxHands = maxHands
    self.modelComplex = modelComplexity
    self.detectionCon = detectionCon
    self.trackCon = trackCon

    self.mpHands = mp.solutions.hands
    self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
    self.mpDraw = mp.solutions.drawing_utils

  def findHands(self, img, draw = True):
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    self.results = self.hands.process(imgRGB)

    # Print landmarks on hands
    if self.results.multi_hand_landmarks:
      # Extract Information on Each Hands
      for handLms in self.results.multi_hand_landmarks:
        if draw:
          self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

    return img
  
  def findPosition(self, img, handNo=0, draw = True):
    lmList = []

    if self.results.multi_hand_landmarks:
      myHand = self.results.multi_hand_landmarks[handNo]

      for id, lm in enumerate(myHand.landmark):
        h, w, c = img.shape
        cx, cy = int(lm.x * w), int(lm.y * h)
        lmList.append([id, cx, cy])
        # Circles for one of the landmarks
        if draw:
          cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
    
    return lmList


def main():
  # Nescessary for Opening Webcam
  cap = cv2.VideoCapture(1)
  cap.open(0, cv2.CAP_DSHOW)
  
  # Call Object
  detector = handDetector()

  # Frame rate initialize
  pTime = 0
  cTime = 0 

  while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmList = detector.findPosition(img)

    if len(lmList) != 0:
      print(lmList[4])

    # Frames management
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
 
    # Frame rate display
    cv2.putText(img, str(int(fps)), (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 8), 2)

    cv2.imshow("Webcam", img)
    cv2.waitKey(1)



if __name__ == "__main__":
  main()