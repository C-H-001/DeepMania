import sys
import os
# 引入你之前的计算脚本
import sr_calculator 

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_sr.py <path_to_osu_file> [DT/HT]")
        return

    file_path = sys.argv[1]
    mod = sys.argv[2] if len(sys.argv) > 2 else ""

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    try:
        # 调用 calculate 函数
        # 注意：你的函数返回的是 (SR, df_corners) 元组
        sr, _ = sr_calculator.calculate(file_path, mod)
        
        print(f"-" * 30)
        print(f"File: {os.path.basename(file_path)}")
        print(f"Mode: 4K Mania (Assumed)")
        print(f"Mod:  {mod if mod else 'None'}")
        print(f"SR:   {sr:.4f} ★")
        print(f"-" * 30)
        
    except Exception as e:
        print(f"Calculation failed: {e}")

if __name__ == "__main__":
    main()