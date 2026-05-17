void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }
void assume(int cond) { if (!cond) { abort(); } }
int __VERIFIER_nondet_int();
int main() {
int x;
int y;
assume((x >= 0));
assume((x <= 2));
assume((y <= 2));
assume((y >= 0));
while (__VERIFIER_nondet_int()) {
{
(x = (x + 2));
(y = (y + 2));
}
}
if ((x == 4)) {
assert((y != 0));
}
}