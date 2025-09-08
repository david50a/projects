from PIL import Image
import numpy as np
class Image_steganography:
    def __fillMSB(self,inp):
        inp=inp.split("b")[-1]
        inp='0'*(7-len(inp))+inp
        return [int(x) for x in inp]
    def __decrypt_pixel(self,pixels):
        pixels=[str(x%2)for x in pixels]
        bin_repr="".join(pixels)
        return chr(int(bin_repr,2))
    def encrypt_text_in_image(self,image_path,msg,target_path=''):
        img=np.array(Image.open(image_path))
        imgArr=img.flatten()
        msg+='<-END->'
        msgArr=[self.__fillMSB(bin(ord(c)))for c in msg]
        idx=0
        for char in msgArr:
            for bit in char:
                if bit==1:
                    if imgArr[idx]==0:
                        imgArr[idx]=1
                    else:
                        imgArr[idx]= imgArr[idx] if imgArr[idx]%2==1 else imgArr[idx]-1
                else:
                    if imgArr[idx]==255:
                        imgArr[idx]=254
                    else:
                        imgArr[idx]= imgArr[idx] if imgArr[idx]%2==0 else imgArr[idx]+1
                idx+=1
        savePath=target_path
        result=Image.fromarray(np.reshape(imgArr,img.shape))
        result.save(savePath)
        return
    def decrypt_text_in_image(self,image_path):
        img=np.array(Image.open(image_path))
        imgArr=np.array(img).flatten()
        decrypted_msg=''
        for i in range(7,len(imgArr),7):
            decrypted_char=self.__decrypt_pixel(imgArr[i-7:i])
            decrypted_msg+=decrypted_char
            if len(decrypted_msg)>10 and decrypted_msg[-7:]=='<-END->':
                break
        return decrypted_msg[:-7]
x=Image_steganography()
print(x.decrypt_text_in_image("test.png"))