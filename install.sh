# install UniGraspTransformer
pip install -e .
# install pytorch_kinematics
cd pytorch_kinematics
pip install -e .
cd ..
# install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# prepare folders
python prepare.py
# unzip hand_assets
cd dexgrasp
unzip hand_assets.zip

