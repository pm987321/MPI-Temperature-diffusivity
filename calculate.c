#include <stdio.h>
#include "mpi.h"
#include <stdlib.h>
#include <unistd.h>
#include <sys/time.h>
#include <math.h>

#define ROW 226
#define COL 8
#define MASTER 0
#define tag 0

void top(double array[],double topTen[],int index[]);
int isHexNum(char c);

int vector[8]={0,3668,7924,0,0,17835,0,0};//特征向量

main( int argc, char *argv[]) 
{ 
  int rank;//进程的id 
  int size;//进程数
  int numworkers=0;//计算进程数
  int source=0;//源头
  int dest=1;//终点
  int averow;//每个进程平均算的行数
  int extra;//多出来的行数
  int rows;//每个工作进程实际收到的层数
  int offset;//偏移量
  int i,j,rc;
  double midResult=0;//存储两个向量相乘的中间结果
  double lenOfVector=0;//存储计算单个向量长度时的中间结果
  double cos;//特征向量的cos值
  MPI_Status status;

  double topTen[10]={-2};//cos值最大的10个
  int matrix[ROW][COL];//特征矩阵
  double static start,end,time;//时间

  double lentmp = 0;
  double len=0;//特征向量的长度
  for(i=0;i<COL;i++)
    lentmp+=vector[i]*vector[i];
  len=sqrt(lentmp);

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);//获取进程的rank
  MPI_Comm_size(MPI_COMM_WORLD,&size);//获取进程的数目

  if(size<2) //至少需要一个主进程和一个计算进程
  {
    printf("至少需要两个进程，退出......\n");
    MPI_Abort(MPI_COMM_WORLD, rc);//终止MPI程序，用于出错处理
  }
  numworkers = size-1;//工作进程数

  start = MPI_Wtime();

  /***************主进程部分***********************/
  if(rank==MASTER)//主进程先进行初始化
  {
    printf("主进程启动...\n");

    //将二进制文件里的数据读到内存保存在matrix数组中
    int data;
    char ch[4];
    i=0,j=0;
    FILE *fp=fopen("test.bin","r");
    if(!fp)
    {
      printf("can't open file\n");
      return -1;
    }

    int count=0;
    while(fscanf(fp, "%s", ch) != EOF)
    {
      data=chToInt(ch);
      if(count==8)
      {
	i++;
        j=0;
	count=0;
      }
      matrix[i][j]=data;
      j++;
      count++;
    }
    fclose(fp);

    averow=ROW/numworkers;
    extra=ROW%numworkers;

    offset=0;
    for(dest=1;dest<=numworkers;dest++)
    {
      if(dest<=extra)
        rows=averow+1;
      else
	rows=averow;

      //将数据发送给工作进程
      MPI_Send(&offset, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
      MPI_Send(&rows, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
      MPI_Send(&matrix[offset][0], rows*COL, MPI_INT, dest, tag,MPI_COMM_WORLD);
      offset=offset+rows;
    }

printf("主进程发送数据完毕\n");

    //从工作进程接收计算结果,保存在tmpTop[ROW]数组中
    double tmpTop[ROW];
    double topTen[10];
    int index[10];
    for(i=1;i<=3;i++)
    {
      MPI_Recv(&offset, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
      MPI_Recv(&rows, 1, MPI_INT, i, tag, MPI_COMM_WORLD, &status);
      MPI_Recv(&tmpTop[offset], rows, MPI_DOUBLE, i, tag,MPI_COMM_WORLD, &status);
    }

printf("主进程接收数据结果完毕\n");

    top(tmpTop,topTen,index);

    for(i=0;i<10;i++)
    {
      printf("%d  ",index[i]);
    }
    printf("\n");

    for(i=0;i<10;i++)
    {
      printf("%f  ",topTen[i]);
    }
    printf("\n");
  }

  /***************工作进程部分*********************/
  else
  {
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    printf("%d进程启动...\n",rank);
    //从主进程接收数据
    MPI_Recv(&offset, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&rows, 1, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);
    MPI_Recv(&matrix[offset][0], rows*COL, MPI_INT, MASTER, tag, MPI_COMM_WORLD, &status);

    printf("%d进程接收数据完毕\n",rank);

    //计算特征向量与矩阵中每行的cos值，并将cos值保存在result数组中
    double result[rows];
    int count=0;
    int flag;
    for(i=offset;i<rows+offset;i++)
    {
      midResult=0;
      lenOfVector=0;
      flag=0;
      for(j=0;j<COL;j++)
      {
        if(matrix[i][j]>1000)
	{
	  flag=1;
	  break;
	}
      }
      for(j=0;j<COL;j++)
      {
	if(flag==1)
	{
	  midResult+=vector[j]*matrix[i][j]/1000;
          lenOfVector+=matrix[i][j]/1000*matrix[i][j]/1000;
	}
	else
	{
	  midResult+=vector[j]*matrix[i][j];
          lenOfVector+=matrix[i][j]*matrix[i][j];
	}
      }
      if(len!=0 && lenOfVector!=0)
      {
        cos=midResult/(len*sqrt(lenOfVector));
        result[count]=cos;
      }
      else
        result[count]=-1;

      count++;
    }
    
    //将计算结果发送到主进程
    MPI_Send(&offset, 1, MPI_INT, MASTER, tag,MPI_COMM_WORLD);
    MPI_Send(&rows, 1, MPI_INT, MASTER, tag,MPI_COMM_WORLD);
    MPI_Send(&result[0], rows, MPI_DOUBLE, MASTER, tag,MPI_COMM_WORLD);

printf("%d进程发送数据到主进程完毕\n",rank);
  }

  /*******************进程结束****************************/

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);//获取进程的rank
  end= MPI_Wtime();

  time=end-start;
  //printf("time=\n");
  //printf("计算完成\n");

  printf("进程%d结束...\n",rank);
  MPI_Finalize(); 
}

//将十六进制（字符数组类型）转化为十进制（int型）
int chToInt(char c[])
{
  int i,tmp,result=0;
  for(i=0;i<4;i++)
  {
    if(c[i]=='0')
      tmp=0;
    else if(c[i]=='1')
      tmp=1;
    else if(c[i]=='2')
      tmp=2;
    else if(c[i]=='3')
      tmp=3;
    else if(c[i]=='4')
      tmp=4;
    else if(c[i]=='5')
      tmp=5;
    else if(c[i]=='6')
      tmp=6;
    else if(c[i]=='7')
      tmp=7;
    else if(c[i]=='8')
      tmp=8;
    else if(c[i]=='9')
      tmp=9;
    else if(c[i]=='a')
      tmp=10;
    else if(c[i]=='b')
      tmp=11;
    else if(c[i]=='c')
      tmp=12;
    else if(c[i]=='d')
      tmp=13;
    else if(c[i]=='e')
      tmp=14;
    else if(c[i]=='f')
      tmp=15;
    //printf("%d",tmp);
    result+=tmp*powInt(16,3-i);
  }
  return result;
}

//整数乘方
int powInt(int a,int n)
{
  if(n==0)
    return 1;
  int result=1;
  while(n!=0)
  {
    result*=a;
    n--;
  }
  return result;
}

//找到数组中十个最大的数，保存值和索引
void top(double array[],double topTen[],int index[])
{
  //int length=sizeof(array)/sizeof(array[0]);
//printf("%d\n",length);
  int j,i;
  for(j=0;j<10;j++)
  {
    double max=-1;
    int local=-1;
    for(i=0;i<ROW;i++)
    {
//printf("%f  ",array[i]);
      if(array[i]>max)
      {
         max=array[i];
         local=i;
      }
    }
    topTen[j]=max;
    index[j]=local;
    array[local]=-2;
//printf("\n");
  }
}