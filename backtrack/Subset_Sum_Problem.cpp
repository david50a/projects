#include <iostream>
using namespace std;
bool find(int arr[], int N, int sum, int current, int index) {
    // Base case: if the current sum is equal to the target sum
    if (sum == current) {
        return true;
    }
    // If we've exceeded the sum or we've gone through all elements
    if (current > sum || index >= N) {
        return false;
    }
    
    // Option 1: Include the current element in the sum
    if (find(arr, N, sum, current + arr[index], index + 1)) {
        return true;
    }
    
    // Option 2: Exclude the current element and move to the next element
    if (find(arr, N, sum, current, index + 1)) {
        return true;
    }
    
    // If neither option works, return false
    return false;
}
int main(){
    int arr[]={1,6,9,6,8,3},sum;
    cout<<"Enter a sum"<<endl;
    cin>>sum;
    int size=sizeof(arr)/sizeof(arr[0]);
    if(find(arr,size,sum,0,0)){cout<<"the combinition has been found"<<endl;}
    else{cout<<"the combinition has not been found"<<endl;}
    return 0;
}