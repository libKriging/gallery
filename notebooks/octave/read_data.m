function A = read_data(name, stem)
  A = dlmread(fullfile("..", "..", "data", [name "_" stem ".csv"]), ",");
end
