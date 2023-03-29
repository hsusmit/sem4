## OS Home Assignment 10

---

_Write a shell script to make a password based menu-driven program, which will give a maximum of three chances to enter the password. If the given password is correct then the program will show the_ <br>
_1. Number of users currently logged in._<br>
_2. Calendar of current month._<br>
_3. Date in the format: dd/mm/yyyy._<br>
_4. Quit_<br>
_The menu should be placed approximately in the centre of the screen and should be displayed in bold._

```bash
#!/bin/bash

PASSWORD="password123"
TRIES=3

function authenticate {
    for i in $(seq 1 $TRIES); do
        read -s -p "Enter password: " input_password
        echo
        if [[ "$input_password" == "$PASSWORD" ]]; then
            return 0
        else
            echo "Incorrect password. Try again."
        fi
    done
    echo "Maximum number of tries exceeded. Exiting."
    exit 1
}

function show_menu {
    clear
    tput bold
    tput cup 5 30
    echo "MENU"
    tput sgr0
    tput cup 6 30
    echo "1. Number of users currently logged in"
    tput cup 7 30
    echo "2. Calendar of current month"
    tput cup 8 30
    echo "3. Date in the format: dd/mm/yyyy"
    tput cup 9 30
    echo "4. Quit"
    tput bold
}

function show_users {
    who | wc -l
}

function show_calendar {
    cal
}

function show_date {
    date +"%d/%m/%Y"
}

function main {
    authenticate
    while true; do
        show_menu
        read -p "Enter your choice: " choice
        case "$choice" in
            1) show_users ;;
            2) show_calendar ;;
            3) show_date ;;
            4) exit 0 ;;
            *) echo "Invalid choice. Try again." ;;
        esac
        read -p "Press Enter to continue"
    done
}
main

```
