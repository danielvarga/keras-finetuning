import sys
import os
import shutil
from collections import defaultdict

def main():
    files_filename, faces_filename, input_dir, output_dir = sys.argv[1:]
    files = [l.strip().split("\t") for l in file(files_filename)]
    faces = [l.strip().split("\t") for l in file(faces_filename)]
    persons = set(l[2] for l in faces)
    assert "unknown" not in persons
    print "list of people:"
    print "\n".join(sorted(persons))
    print len(files), "files found"
    files = [(indx, filename) for (indx, filename) in files if filename.lower().endswith((".jpg", ".png"))]
    print len(files), "image files kept"
    files_dict = {l[0]:l[1] for l in files}
    faces_dict = {l[0]:l[2] for l in faces}
    common = sorted(set(files_dict.keys()) & set(faces_dict.keys()))
    print len(common), "images with face information"
    d = defaultdict(list)
    for (indx, filename) in files:
        person = faces_dict.get(indx, "unknown")
        d[person].append(filename)
    for p in sorted(persons):
        print p, len(d[p])

    def normalized(p):
        p = p.replace("/", "-")
        p = p.replace(" ", "-")
        return p

    os.mkdir(output_dir)
    for p in persons:
        os.mkdir(os.path.join(output_dir, normalized(p)))
    for p in persons:
        for filename in d[p]:
            shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, normalized(p), filename))

main()
