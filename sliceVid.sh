# get duration of video from input argument 1 (the path to the video)
n=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $1)

# round duration of video
n=$(echo "$n/1" | bc)

# set parameters, extract filenames and paths, make a save directory
rootdir=$(echo $(dirname $1))
vid=$(echo $(basename $1))
vid=$(echo $vid | cut -d"." -f1)
savedir="$rootdir/$vid"
mkdir $savedir

# slice into 5 second slices
start=0
duration=5
while test $start -lt $n
do
    filename="${vid}_slice${start}.mp4"
    filepath="$savedir/$filename"
    ffmpeg -i $1 -ss $start -t $duration $filepath
    start=$(($start+$duration))
done
