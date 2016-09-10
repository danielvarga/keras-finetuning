#!/bin/bash

if [ -z "$1" ] || [ -z "$2" ] ; then echo "Usage: $0 PATH_TO_APPLE_PHOTOS_WORKING_DIR OUTPUT_DIR"; exit -1 ; fi

PHOTOSPATH=$1
OUTPUT_DIR=$2

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

sqlite3 -separator $'\t' $1/database/ImageProxies.apdb "select modelId, resourceUuid, filename from RKModelResource order by modelId;" | awk 'BEGIN{FS="\t"; for(n=0;n<256;n++)ord[sprintf("%c",n)]=n}  { print $1 "\t" ord[substr($2,1,1)] "/" ord[substr($2,2,1)] "/" $2 "/" $3 }' > imagefiles
sqlite3 -separator $'\t' $1/database/Person.db "select f.modelId,f.personId,p.name from RKFace f join RKPerson p on f.personId=p.modelId order by f.modelId;" > faces

python $DIR/collect_apple_photos.py imagefiles faces $PHOTOSPATH/resources/modelresources $OUTPUT_DIR
