# input folder location of vids
files=$(echo $(ls $1/*.mp4))
for file in $files
do
    vid=$(echo $(basename $file))
    vid=$(echo $vid | cut -d"." -f1)
    mkdir -p $1/scaled
    filename="$1/scaled/${vid}_scaled.mp4"
    ffmpeg -i $file -vf scale=224:224 -an $filename
done
