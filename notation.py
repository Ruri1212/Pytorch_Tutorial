import matplotlib.pyplot as plt
fig, axs = plt.subplots(2, 3, figsize=(10, 5))
print(fig,axs)


## zip関数
word = ["ad","uta"]
num = [1,2]

for w,n in zip(word,num):
    print(w,n)

## nextとイテレータ
li = [1,2,3,4]
it = iter(li)
while True:
    try:
        print(next(it))
    except StopIteration:
        break