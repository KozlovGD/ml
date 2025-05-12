
f = open('train.csv','r')
kol = 0
ind = 1
for line in f:
   #print(line)
   kol = kol + 1
#print("\n")
print("kolichestvo strok",kol)
kol = kol - 1;
f.close()


f0 = open('train.csv','r')
f1 = open('train_1.csv','w')
f2 = open('train_2.csv','w')
f3 = open('val_1.csv','w')
f4 = open('val_2.csv','w')

for line in f0:
    listinf = line.split(',')
    if (listinf[0] == 'PassengerId'):
       f1.write(line)
       f2.write(line)
       f3.write(line)
       f4.write(line)
       continue
    if (ind <= kol/2):
        f1.write(line)
        f4.write(line)
    else:
        f3.write(line)
        f2.write(line)
    ind = ind + 1
        
f0.close()
f1.close()
f2.close()
f3.close()    
