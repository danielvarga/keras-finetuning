sqlite3 -separator $'\t' ImageProxies.apdb "select modelId, resourceUuid, filename from RKModelResource order by modelId;" | awk 'BEGIN{FS="\t"; for(n=0;n<256;n++)ord[sprintf("%c",n)]=n}  { print $1 "\t" ord[substr($2,1,1)] "/" ord[substr($2,2,1)] "/" $2 "/" $3 }' > filenames
sqlite3 -separator $'\t' Person.db "select f.modelId,f.personId,p.name from RKFace f join RKPerson p on f.personId=p.modelId order by f.modelId;" > faces

