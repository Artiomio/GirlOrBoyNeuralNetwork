#!/bin/bash
javac BoyOrGirlRobot.java -encoding UTF-8 && java BoyOrGirlRobot 2> error.log 1> out.txt || msgbox Error!
