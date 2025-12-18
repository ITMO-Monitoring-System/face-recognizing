from face_service.core.persons_loader import PersonsSource, load_persons
from face_service.core.recognize import recognize_image

if __name__ == "__main__":
    persons = load_persons(PersonsSource(mode="json"))
    print("Loaded persons:", len(persons))

    # пример 
    test_img = r"D:\FaceRecognizing\FaceRecgnizingProj\faces\img_2.png"
    print(recognize_image(test_img, persons, threshold=0.45))
