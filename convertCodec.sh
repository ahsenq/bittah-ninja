avi="$1/*.avi"
num_avi=$(ls $avi | wc -l)
mp4="$1/*.mp4"
num_mp4=$(ls $mp4 | wc -l)

# convert bad mp4 to good mp4
i=0
for f in $mp4
do
    root=$(echo $(dirname $f))
    base=$(echo $(basename $f))
    vid=$(echo $base | cut -d"." -f1)
    filename="$root/$vid.avi"
    ffmpeg -hide_banner -loglevel panic -i $f $filename -y
    ffmpeg -hide_banner -loglevel panic -i $filename $f -y
    rm $filename
    i=$(($i+1))
    echo -n "converting codecs... $i/$num_mp4\r"
done

# convert avi to mp4
i=0
for f in $avi
do
    root=$(echo $(dirname $f))
    base=$(echo $(basename $f))
    vid=$(echo $base | cut -d"." -f1)
    filename="$root/$vid.mp4"
    ffmpeg -hide_banner -loglevel panic -i $f $filename -y
    rm $f
    i=$(($i+1))
    echo -n "converting codecs... $i/$num_avi\r"
done
