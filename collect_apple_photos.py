import sys
import os
import shutil
import unicodedata
from collections import defaultdict


UNKNOWN = "unknown"

def remove_accents(input_unicode):
    nfkd_form = unicodedata.normalize('NFKD', input_unicode)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

def normalized(p):
    p = p.replace("/", "-")
    p = p.replace(" ", "-")
    return remove_accents(p.decode("utf-8")).encode("ascii")


def main():
    files_filename, faces_filename, input_dir, output_dir = sys.argv[1:]
    files = [l.strip().split("\t") for l in file(files_filename)]
    faces = [l.strip().split("\t") for l in file(faces_filename)]
    persons = set(l[2] for l in faces)
    assert UNKNOWN not in persons
    print len(files), "files found"
    files = [(indx, filename) for (indx, filename) in files if filename.lower().endswith((".jpg", ".png"))]
    print len(files), "image files kept"
    files_dict = {l[0]:l[1] for l in files}
    faces_dict = {l[0]:l[2] for l in faces}
    common = sorted(set(files_dict.keys()) & set(faces_dict.keys()))
    print len(common), "images with face information"
    d = defaultdict(list)
    for (indx, filename) in files:
        person = faces_dict.get(indx, UNKNOWN)
        d[person].append(filename)
    if UNKNOWN in d.keys():
        persons.add(UNKNOWN)
    print
    print "images found for each person:"
    for p in sorted(persons):
        print p, len(d[p])

    try:
        os.mkdir(output_dir)
    except:
        print "please remove the output directory"
        raise
    for p in persons:
        os.mkdir(os.path.join(output_dir, normalized(p)))
    for p in persons:
        for filename in d[p]:
            c1, c2, uu, f = filename.split("/")
            generated_filename = uu+"-"+f
            shutil.copy(os.path.join(input_dir, filename), os.path.join(output_dir, normalized(p), generated_filename))


main()
