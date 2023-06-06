import streamlit as st

def add_numbers(a, b):
    return a + b

def main():
    st.title("Addition App")
    st.write("Enter two numbers and click 'Add' to calculate their sum.")

    number1 = st.number_input("Enter the first number", value=0)
    number2 = st.number_input("Enter the second number", value=0)

    if st.button("Add"):
        result = add_numbers(number1, number2)
        st.success(f"The sum of {number1} and {number2} is {result}.")

if __name__ == "__main__":
    main()