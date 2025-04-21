import sysconfig
import sys
print("Executable:", sys.executable)
print("Include dir:", sysconfig.get_path("include"))
print("Lib dir (look for .lib):", sysconfig.get_config_var("LIBDIR"))