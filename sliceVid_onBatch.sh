# get duration of video from input argument 1 (the path to the video)
files=$(echo $(ls $1))
for file in $files
do
    n=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $1$file)

    # round duration of video
    n=$(echo "$n/1" | bc)

    # set parameters, extract filenames and paths, make a save directory
    vid=$(echo $(basename $1))
    vid=$(echo $vid | cut -d"." -f1)

    # slice into 5 second slices
    start=0
    duration=5
    while test $start -lt $n
    do
        filename="$1${vid}_slice${start}.mp4"
        ffmpeg -i $1$file -ss $start -t $duration $filename
        start=$(($start+$duration))
    done
    rm $1$file
done
