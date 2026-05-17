void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }
void assume(int cond) { if (!cond) { abort(); } }
int __VERIFIER_nondet_int();
int main() {
int i;
int x;
int y;
(i = 0);
assume((x >= 0));
assume((y >= 0));
assume((x >= y));
while (__VERIFIER_nondet_int()) {
if ((i < y)) {
(i = (i + 1));
}
}
if ((i >= x)) {
if ((0 > i)) {
assert((i >= y));
}
}
}