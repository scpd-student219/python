import cv2

class Face_Detector:
    def __init__(self, template, image_path):
        self.face_cascade=cv2.CascadeClassifier(template)
        self.image = cv2.imread(image_path)
        self.gray_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
    def detect_faces(self):
        self.faces=self.face_cascade.detectMultiScale(self.gray_image, scaleFactor=1.1, minNeighbors=5)
    def draw_rectangles(self):
        for x,y,w,h in self.faces:
            self.image = cv2.rectangle(self.image, (x,y), (x+w,y+h), (0, 255), 3)
    def display_image(self):
        self.resized=cv2.resize(self.image, (int(self.image.shape[1]/3), int(self.image.shape[0]/3)))
        cv2.imshow("Faces", self.resized)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

finder = Face_Detector("haarcascade_frontalface_default.xml", "news.jpg")

finder.detect_faces()
finder.draw_rectangles()
finder.display_image()
