def find_wrong_chars(annotation, prediction):
    wrong_chars = []

    for char in annotation:
        annotation_index = annotation.index(char)
        
        if char in prediction and prediction.index(char) != annotation_index:
            wrong_chars.append(char)

    return wrong_chars


print(find_wrong_chars("ABC1234", "A8C1235"))