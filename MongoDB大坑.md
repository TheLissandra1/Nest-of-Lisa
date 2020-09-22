# Nest-of-Lisa
老折磨了MongoDB，mlgbd
写一个安装和使用中的各种大坑

首先官网下载mongoDB客户端（这只是一个环境），我建议用安装包的时候不要安装compass，否则会装很久，请后续自行官网下载MongoDB Compass（可视化和管理数据库） 和 MongoDB Database Tools （各种好用的工具包）安装，一定要装这两个，否则可视化和导入数据烦死了艹艹艹

大坑一：
    4.4版本之后，安装在系统盘一般没事，自定义安装在其他盘里会有权限问题，直接ignore，装完。你可能会发现有点问题（嘿嘿），在安装mongodb的文件夹下的\data目录下新建db文件夹，
    然后打开cmd，输入mongod --dbpath 路径（如D:\MongoDB\data\db),回车，如果在某一时间卡住了，不用管，打开另一个新的cmd，输入mongo。
    你可能会发现无法启动服务，此时参考这个教程 https://blog.csdn.net/qq_20084101/article/details/82261195   如果还不好使我也没法子
  
大坑二：
    安装好了，下次再用，要先开一个cmd启动服务，再开一个cmd输入mongo，两个cmd，才能使用
    
大坑三：
    如何安装 MongoDB Database Tools，下载，解压，把一堆exe文件扔进安装mongodb的bin文件夹下，和其他.exe文件呆在一起即可。
    
大坑四：
    如何使用mongoimport导入csv数据，参考官方文档即可
    https://docs.mongodb.com/database-tools/mongoimport/
    重点：使用mongoimport是在cmd环境下，并且要开启第一个mongod的服务，然后再打开第二个cmd调用mongoimport,不要输mongo，否则会报错
    
大坑五：
    新建数据库，如果数据库里没有一条数据，该数据库将无法通过show dbs显示
