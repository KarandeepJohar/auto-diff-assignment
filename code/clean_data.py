import sys

KEEP = ['Organisation','Work','Species','Person','Place']

fin = sys.argv[1]
fout = fin+'.clean'

# filter
inp = open(fin)
op = open(fout,'w')
for line in inp:
    ent,cl = line.rstrip().split('\t')
    if cl in KEEP:
        op.write('%s\t%s\n' % (ent,cl))
op.close()
inp.close()
