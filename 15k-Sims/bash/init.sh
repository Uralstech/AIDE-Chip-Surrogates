# RUN THIS BEFOREHAND
# sudo apt update
# sudo apt install -y build-essential git m4 scons zlib1g zlib1g-dev \
#     libprotobuf-dev protobuf-compiler libprotoc-dev libgoogle-perftools-dev \
#     python3-dev libboost-all-dev pkg-config python3-tk clang-format-15 python3-pip gcc-riscv64-linux-gnu
#
# pip install -r requirements.txt
# 
# screen -S gem5run # Optional

git clone https://github.com/gem5/gem5
cd gem5

git checkout stable
scons build/RISCV/gem5.opt -j $(nproc)