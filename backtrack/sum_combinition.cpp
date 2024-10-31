#include <iostream>
using namespace std;
bool findCombination(int arr[], int size, int index, int sum, int current, int numbers[], int& count) {
    if (current == sum) {
        return true;
    }
    if (size <= index || current > sum) {
        return false;
    }
    if (findCombination(arr, size, index + 1, sum, current + arr[index], numbers, count)) {
        numbers[count++] = arr[index];
        return true;
    }
    if (findCombination(arr, size, index + 1, sum, current, numbers, count)) {
        return true;
    }

    return false;
}

int main(){
    int arr[]={1,2,6,7,9,8,3,8,5,3,4};
    int size=0,sum;
    int count=0,total;
    cin>>sum;
    for(int i=0;i<sizeof(arr)/sizeof(arr[0]);i++){
        int combinition[sizeof(arr)/sizeof(arr[0])]={0};
        if(findCombination(arr,sizeof(arr)/sizeof(arr[0]),i,sum,0,combinition,count)){
            cout<<"found the combinition"<<endl<<"the combination is: ";}
            for(int j=0;j<sizeof(arr)/sizeof(arr[0]);j++){if(combinition[j]!=0)cout<<combinition[j]<<" ";}
            cout<<endl;
        count=0;
        total=0;
    }
    return 0;
}