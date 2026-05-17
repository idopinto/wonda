void assert(int cond) { if (!(cond)) { ERROR : { reach_error(); abort(); } } }
void assume(int cond) { if (!cond) { abort(); } }
int __VERIFIER_nondet_int();
int main() {
int lock;
int x;
int y;
(y = (x + 1));
(lock = 0);
while ((x != y)) {
{
if (__VERIFIER_nondet_int()) {
{
(lock = 1);
(x = y);
}
} else {
{
(lock = 0);
(x = y);
(y = (y + 1));
}
}
}
}
assert((lock == 1));
}