from utils.io_utils import single_file, folder_image

def main():
    print("=== Image Processing Menu ===")
    print("1. Process a single image")
    print("2. Process all images in a folder")
    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "1":
        single_file()
    elif choice == "2":
        folder_image()
    else:
        print("❌ Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
