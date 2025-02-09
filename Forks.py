import os

print("Before Fork\n")

pid1 = os.fork() # creates 1 child and 1 parent 
pid2 = os.fork() # createes 1 child and 1 parent for (pid1) 

if pid1 == 0:
    print(f"Child : {os.getpid()} , Parent1 : {os.getppid()}")
elif pid1 > 0:
    print(f"Parent1 : {os.getpid()} , Child : {pid1}")
if pid2 == 0:
    print(f"Child : {os.getpid()} , Parent2 : {os.getppid()}")
elif pid2 > 0:
    print(f"Parent2 : {os.getpid()} , Child : {pid2}")
else:
    print("Creation Of Fork Fails")
    
print("After Fork\n")