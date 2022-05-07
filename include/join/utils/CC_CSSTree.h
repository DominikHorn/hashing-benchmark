#pragma once

/* An adapted variation of CSSTree from "Relational Joins on Graphics Processors" paper, sent by the original authors */

#include "config.h"            /* autoconf header */
#include "configs/base_configs.h"
#include "configs/eth_configs.h"

#include "utils/eth_data_structures.h"
#include "utils/data_generation.h"
#include "utils/eth_generic_task_queue.h"
#include "utils/base_utils.h"
#include "utils/math.h"
#include "utils/memory.h"


using namespace std;

//we use implicit pointer to perform the addressing.
template<typename KeyType, typename PayloadType>
class CC_CSSTree
{
public:
	uint64_t numRecord;
	Tuple<KeyType, PayloadType> *data;
	//we use the BFS layout as the default layout.
	int numNode;
	int level;
	int gResult;
	KeyType *ntree;
	int fanout;
	int blockSize;
	int *vStart;
	int *vG;//vG[0] is used in computing the position for level 1.
	int numKey;
	CC_CSSTree(Tuple<KeyType, PayloadType> *d, uint64_t numR, int f)
	{
        data=d;
		numRecord=numR;	

		fanout=f;
		blockSize=fanout-1;
		int numLeaf=divRoundUp(numR,blockSize);		
		level=1;
		int temp=numLeaf;
		while(temp>1)
		{
			temp=divRoundUp(temp, fanout);
			level++;
		}
		numNode=(int)((pow((double)fanout,(double)level)-1)/(fanout-1));
		numKey=numNode*blockSize;
		//ntree=new KeyType[numKey];
        ntree= (KeyType *) malloc(sizeof(KeyType) * numKey);
		//vStart=new int[level];
        vStart= (int *) malloc(sizeof(int) * level);        
		//vG=new int[level];
        vG= (int *) malloc(sizeof(int) * level);

//#ifdef DEBUG
//		cout<<numLeaf<<","<<level<<", "<<numNode<<endl;
//#endif
		//layout the tree from bottom up.
		int i=0,j=0,k=0;
		int startNode=0;
		int endNode=0;
		int startKey, endKey;
		int curIndex;
		for(i=0;i<numNode;i++)
			ntree[i]=-1;
		//for <level-1>, i.e., the leaf level. [start,end]
		for(i=0;i<level;i++)//level
		{
			startNode=(int)((pow((double)fanout,(double)i)-1)/(fanout-1));
			endNode=(int)((pow((double)fanout,(double)(i+1))-1)/(fanout-1));
			for(j= startNode;j< endNode;j++)//which node
			{
				startKey=j*blockSize;
				endKey=startKey+blockSize;
				for(k=startKey;k<endKey;k++)
				{
					curIndex=(int)(blockSize*pow((double)fanout,(double)(level-i-1))*(k+1-startNode*blockSize+(j-startNode))-1);
					if(curIndex<numRecord+blockSize)
					{
						if(curIndex>=numRecord)
							curIndex=numRecord-1;
						ntree[k]=data[curIndex].key;
					}
					else
						break;
				}
			}
		}
	}
	
    ~CC_CSSTree() {}

    //return the start position of searching the key.
    int search(KeyType key)
    {
        int i=0;
        int curIndex=0;
        int curNode=0;
        int j=0;
        int oldJ=0;
        //search
        for(i=0;i<level;i++)
        {
            for(j=0;j<blockSize;j++)
            {
                if(ntree[curIndex+j]==-1)
                    break;
                if(key<=ntree[curIndex+j])
                    break;
            }
            curNode=(fanout*(curNode)+j+1);
            curIndex=curNode*blockSize;
    //#ifdef DEBUG
    //		cout<<curNode<<", "<<j<<", "<<ntree[curIndex]<<";   ";
    //#endif
        }
        curIndex=(curNode-numNode)*blockSize;
        if(curIndex>numRecord) curIndex=numRecord-1;
        //cout<<"I: "<<curIndex<<", ";//cout<<endl;
        return curIndex;
    }

    void print()
	{
		int i=0, j=0;
		int k=0;
		int startNode=0;
		int endNode=0;
		int startKey, endKey;
		for(i=0;i<level;i++)//level
		{
			cout<<"Level, "<<i<<endl;
			startNode=(int)((pow((double)fanout,(double)i)-1)/(fanout-1));
			endNode=(int)((pow((double)fanout,(double)(i+1))-1)/(fanout-1));
			for(j= startNode;j< endNode;j++)//which node
			{
				cout<<"Level, "<<i<<", Node, "<<j<<": ";
				startKey=j*blockSize;
				endKey=startKey+blockSize;
				for(k=startKey;k<endKey;k++)
				{
					cout<<ntree[k]<<", ";	
				}
				cout<<endl;
			}
		}
		for(i=0;i<numRecord;i++)
		{
			cout<<data[i].key<<", ";
			if(i%(fanout-1)==(fanout-2))
			cout<<"*"<<endl;
		}
	}

};

template<typename KeyType, typename PayloadType>
int cc_constructCSSTree(Tuple<KeyType, PayloadType> *Rin, int rLen, CC_CSSTree<KeyType, PayloadType> **tree)
{
	*tree=new CC_CSSTree<KeyType, PayloadType>(Rin,rLen,INLJ_CSS_TREE_FANOUT);
	return (*tree)->numNode;
}

/*template<typename KeyType, typename PayloadType>
int cc_equiTreeSearch(Tuple<KeyType, PayloadType> *Rin, int rLen, CC_CSSTree<KeyType, PayloadType> *tree, KeyType keyForSearch, Tuple<KeyType, PayloadType>** Rout)
{
	int result=0;
	int i=0;
	int curIndex=tree->search(keyForSearch);
	cout<<curIndex<<", ";
	LinkedList<KeyType, PayloadType> *ll=(LinkedList<KeyType, PayloadType>*)malloc(sizeof(LinkedList<KeyType, PayloadType>));
	ll->init();
	for(i=curIndex-1;i>0;i--)
		if(Rin[i].key==keyForSearch)
		{
			ll->fill(Rin+i);
			result++;
		}
		else
			if(Rin[i].key<keyForSearch)
			break;
	for(i=curIndex;i<rLen;i++)
		if(Rin[i].key==keyForSearch)
		{
			ll->fill(Rin+i);
			result++;
		}
		else
			if(Rin[i].key>keyForSearch)
			break;
	(*Rout)=(Tuple<KeyType, PayloadType> *)malloc(sizeof(Tuple<KeyType, PayloadType>)*result);
	ll->copyToArray((*Rout));
	delete ll;
	return result;
}*/
