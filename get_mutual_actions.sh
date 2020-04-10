#! /bin/bash

mkdir $2

for ((i=50;i<61;i+=1));
do
	cp $1/*A0$i.skeleton $2/
done

for ((i=106;i<121;i+=1));
do
	cp $1/*A$i.skeleton $2/
done

echo "Totally $(ls $2 | wc -l) files copied"
