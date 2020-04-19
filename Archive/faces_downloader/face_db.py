import vk
import time
from vk.exceptions import VkException, VkAPIError
import requests
import random

TOKEN_FILE_NAME = "token.txt"

def token_from_file():
    with open(TOKEN_FILE_NAME, "r") as myfile:
        token = myfile.readlines()[0]
        return token



       
token = token_from_file()
vkapi = vk.API(vk.Session(access_token = token), v='5.38')




vk_execute_script = """
    var id_list = %s;
    var res = [];

    var i = 0;
    var c;
    var sex;
    var date;

    while (i < id_list.length) {
        var a= API.users.get({"user_ids":  id_list[i], "fields" : ["photo_id", "sex", "bdate"]});
        var sex_ = a[0]["sex"];
        sex = "undefined";

        if (sex_ == 2) 
              sex = "male";

        if (sex_ == 1) 
              sex = "female";

        date = a[0]["bdate"];
        var photo_id = a[0]["photo_id"];
        c = API.photos.getById({"photos" : photo_id})@.photo_807[0];
        res = res +[{"id":id_list[i], "photo": c, "sex": sex, "bdate": date}];
        i = i + 1;
    }

return res;    """ 


while True:
    try:
        time.sleep(1)
        random_list = [random.randint(1000, 20000000) for x in range(10)]
        resp = vkapi.execute(code=(vk_execute_script % str(random_list)))
        for i in resp:
            if i["photo"] and i["sex"]!="undefined": # if photo is present
                print(i)
                if i["sex"]=="female":
                    file_name = r"o:\faces\female\%d.jpg" % i["id"]
                else:
                    file_name = r"o:\faces\male\%d.jpg" % i["id"]
                image_data = requests.get(i["photo"], timeout=5, allow_redirects=True).content
                with open(file_name, "wb") as f: 
                    f.write(image_data)


    except VkAPIError as e:
        print(e)
        time.sleep(3)

    except Exception as e:
        print("Error!: ", e)
        time.sleep(3)
