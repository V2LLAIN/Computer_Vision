## 📌 Cityscapes(19+1 classes) Dataset Downloads link:
##### (https://drive.google.com/file/d/1eff6XpWRNRL-nVQxXvqrT6_e-EWVXg5d/view?usp=sharing)

### Move to your Server:
in my case, host name is root

    scp -P 20163 cityscapes.zip root@서버주소:/root/Study/Computer Vision/

#
### How to Train? 
    nohup python3 train.py &
#### (you can check in nohup.out)
#### (you can check by using
    jobs
[1]+  Running                 nohup python3 train.py &
#### then you can see console like above
