import sys

if len(sys.argv) != 2:
    print('Usage:')
    print('python make_video_txt.py K(integer)')
    exit()

allframes = 'train.txt'
keyframes = 'VID_train_15frames.txt'
K = sys.argv[1]
k = int(K)
outputfilename = 'VID_train_K_{:d}.txt'.format(k)

allframesread = open(allframes).read().strip().split()
print('Total number of frames: %d'%len(allframesread))
keyframesread = open(keyframes).read().strip().split()
print('Number of key frames: %d'%len(keyframesread))
inputlist = open(allframes).readlines()

allframespath = allframesread[::2]
folder1 = []
indexinfolder = []
for i in range(len(allframespath)):
    #if not hard code, uncomment and modify
    #indexinfolder.append(allframespath[i].split('/')[-1])
    folder1.append(allframespath[i][:-7])
    indexinfolder.append(allframespath[i][-6:])

folder2 = keyframesread[::4]
keyframeskeyindex = keyframesread[2::4]
keysize = keyframesread[3::4]
for i in range(len(folder2)):
    folder2[i] = folder2[i][6:]
#print('keyframekeyindex: %s'%keyframeskeyindex)

outputlist = []
outputindex = []
videosize = []
for i in range(len(inputlist)):
    print('Progress: %d%%, #%d'%((100*i/len(inputlist)), i))
    print(inputlist[i])

    #indices = [j for j in range(len(folder2)) if folder1[i] == folder2[j]]
    #for j in indices:
    try:
        print('folder1[i]: %s'%folder1[i])
        begin = folder2.index(folder1[i])
    except ValueError:
        break
    else:
        for j in range(begin, begin + 15):
            if (int(keyframeskeyindex[j]) - k) <= int(indexinfolder[i]) and int(indexinfolder[i]) <= (int(keyframeskeyindex[j]) + k):
                outputlist.append(folder1[i])
                outputindex.append(str(int(indexinfolder[i])))
                videosize.append(keysize[j])
                break

outputfile = open(outputfilename, 'w')
for i in range(len(outputlist)):
    outputfile.write('train/' + outputlist[i] + ' 1 ' + outputindex[i] + ' ' + videosize[i] + '\n')