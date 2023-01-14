# -*- coding: utf-8 -*-
def text_create():  #定义一个text_create函数；
    path = 'E:/python/'   #给变量path赋值为路径；
    for text_name in range(1,11):   
    # 将1-10范围内的每个数字依次装入变量text_name中，每次命名一个文件；
        with open (path + str(text_name) + '.txt','w') as text: 
        # 打开位于路径的txt文件，并给每一个text执行写入操作；
            text.write(str(text_name+9)) #给每个文件依次写入；
            text.close()  #关闭文件；
            print('Done')
            
text_create()