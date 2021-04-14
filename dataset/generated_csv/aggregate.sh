#!/bin/sh

rm -f ./../dataset.csv
ls Query* | while read item;
do
    cat $item >> ./../dataset.csv
done


sed -e 's/title,body,tags//g' ./../dataset.csv > ./../dataset_tmp.csv

echo "title,body,tags" > ./../dataset.csv
cat ./../dataset_tmp.csv >> ./../dataset.csv
rm -f ./../dataset_tmp.csv




