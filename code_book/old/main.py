from locallib import input_default_with 
import randomwalk, tictactoe, mab

def main():
    """
    Futher than Tictactoe applications are considered, including random walk, maze, etc. 
    """
    Done = False
    while not Done:
        print()
        print('==RL Aget Testing Framework ==')
        print('- Developed by Sungjin Kim, 2020')
        print()
        print('0) Tictactoe')
        print('1) Random walk')
        print('2) MAB')
        print('999) Quit')
        Q1 = input_default_with('Which application do you want to perform?', 999, int)
        if Q1 == 0:
            tictactoe.main()
        elif Q1 == 1:
            randomwalk.main()
        elif Q1 == 2:
            mab.main()
        elif Q1 == 999:
            Done = True
        else:
            print('Type a different option listed in the above table.')   


if __name__ == "__main__":
    # This is the main function.
    main()
