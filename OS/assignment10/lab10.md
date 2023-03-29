## OS Lab Assignment 10

---

_Devise a menu-driven shell program that accepts values from 1 to 4 and performs action depending upon the number keyed in:_<br>
_1. List of users currently logged in_<br>
_2. Present date_<br>
_3. Present working directory_<br>
_4. Quit_<br>

```bash
#!/bin/bash

while true
do
    echo "Please select an option:"
    echo "1. List of users currently logged in"
    echo "2. Present date"
    echo "3. Present working directory"
    echo "4. Quit"
    read choice

    case $choice in
        1)
            who;;
        2)
            date;;
        3)
            pwd;;
        4)
            exit;;
        *)
            echo "Invalid choice.";;
    esac
done

```
