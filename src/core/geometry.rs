use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

pub type Vector2i = Vector2<i32>;
pub type Vector2f = Vector2<f32>;
pub type Vector3i = Vector3<i32>;
pub type Vector3f = Vector3<f32>;

pub trait NumberField:
    Mul<Output = Self>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Default
    + PartialOrd
    + Neg<Output = Self>
    + Copy
    + Clone
{
}

impl NumberField for i32 {}

impl NumberField for f32 {}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vector2<T> {
    pub x: T,
    pub y: T,
}

impl Vector2<f32> {
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y
    }

    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn normalize(v: &Self) -> Self {
        *v / v.length()
    }
}

impl Vector2<i32> {
    pub fn length_squared(&self) -> i32 {
        self.x * self.x + self.y * self.y
    }

    pub fn length(&self) -> f32 {
        (self.length_squared() as f32).sqrt()
    }

    pub fn normalize(v: &Self) -> Self {
        *v / v.length() as i32
    }
}

impl<T: NumberField> Vector2<T> {
    pub fn new(x: T, y: T) -> Self {
        Vector2 { x, y }
    }

    pub fn abs(v: &Self) -> Self {
        Vector2 {
            x: abs_t(v.x),
            y: abs_t(v.y),
        }
    }

    pub fn dot(v1: &Self, v2: &Self) -> T {
        v1.x * v2.x + v1.y * v2.y
    }

    pub fn abs_dot(v1: &Self, v2: &Self) -> T {
        abs_t(Vector2::dot(v1, v2))
    }

    pub fn min_component(v: &Self) -> T {
        min_t(v.x, v.y)
    }

    pub fn max_component(v: &Self) -> T {
        max_t(v.x, v.y)
    }

    pub fn max_dimensions(v: &Self) -> usize {
        if v.x > v.y {
            0
        } else {
            1
        }
    }

    pub fn permute(v: &Self, x: usize, y: usize) -> Self {
        Vector2 { x: v[x], y: v[y] }
    }
}

impl Div<i32> for Vector2<i32> {
    type Output = Vector2<i32>;

    fn div(self, rhs: i32) -> Self::Output {
        let inv = 1.0 / rhs as f32;
        Vector2 {
            x: (self.x as f32 * inv) as i32,
            y: (self.y as f32 * inv) as i32,
        }
    }
}

impl Div<f32> for Vector2<f32> {
    type Output = Vector2<f32>;

    fn div(self, rhs: f32) -> Self::Output {
        let inv = 1.0 / rhs;
        Vector2 {
            x: self.x * inv,
            y: self.y * inv,
        }
    }
}

impl DivAssign<i32> for Vector2<i32> {
    fn div_assign(&mut self, rhs: i32) {
        self.x = (self.x as f32 / rhs as f32) as i32;
        self.y = (self.y as f32 / rhs as f32) as i32;
    }
}

impl DivAssign<f32> for Vector2<f32> {
    fn div_assign(&mut self, rhs: f32) {
        self.x = self.x / rhs;
        self.y = self.y / rhs;
    }
}

impl<T: NumberField> Add<Vector2<T>> for Vector2<T> {
    type Output = Vector2<T>;

    fn add(self, rhs: Vector2<T>) -> Self::Output {
        Vector2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: NumberField> AddAssign<Vector2<T>> for Vector2<T> {
    fn add_assign(&mut self, rhs: Vector2<T>) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
    }
}

impl<T: NumberField> Sub<Vector2<T>> for Vector2<T> {
    type Output = Vector2<T>;

    fn sub(self, rhs: Vector2<T>) -> Self::Output {
        Vector2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T: NumberField> SubAssign<Vector2<T>> for Vector2<T> {
    fn sub_assign(&mut self, rhs: Vector2<T>) {
        self.x = self.x - rhs.x;
        self.y = self.y - rhs.y;
    }
}

impl<T: NumberField> Neg for Vector2<T> {
    type Output = Vector2<T>;

    fn neg(self) -> Self::Output {
        Vector2 {
            x: -self.x,
            y: -self.y,
        }
    }
}

impl<T: NumberField> Mul<T> for Vector2<T> {
    type Output = Vector2<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Vector2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T: NumberField> MulAssign<T> for Vector2<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x = self.x * rhs;
        self.y = self.y * rhs;
    }
}

impl Mul<Vector2<i32>> for i32 {
    type Output = Vector2<i32>;

    fn mul(self, rhs: Vector2<i32>) -> Self::Output {
        rhs * self
    }
}

impl Mul<Vector2<f32>> for f32 {
    type Output = Vector2<f32>;

    fn mul(self, rhs: Vector2<f32>) -> Self::Output {
        rhs * self
    }
}

impl<T> Index<usize> for Vector2<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("index provided non-existent"),
        }
    }
}

impl<T> IndexMut<usize> for Vector2<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("index provided non-existent"),
        }
    }
}

#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Vector3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl Vector3<f32> {
    pub fn length_squared(&self) -> f32 {
        self.x * self.x + self.y * self.y + self.x * self.y
    }

    pub fn length(&self) -> f32 {
        self.length_squared().sqrt()
    }

    pub fn normalize(v: &Self) -> Self {
        *v / v.length()
    }

    pub fn coordinate_system(from: &Self) -> (Self, Self) {
        let v1 = {
            if abs_t(from.x) > abs_t(from.y) {
                Vector3::new(-from.z, 0.0, from.x) /
                    ((from.x * from.x + from.z * from.z) as f32).sqrt()
            } else {
                Vector3::new(0.0, from.z, -from.y) /
                    ((from.y * from.y + from.z * from.z) as f32).sqrt()
            }
        };
        (v1, Vector3::cross(from, &v1))
    }
}

impl Vector3<i32> {
    pub fn length_squared(&self) -> i32 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn length(&self) -> f32 {
        (self.length_squared() as f32).sqrt()
    }

    pub fn normalize(v: &Self) -> Self {
        *v / v.length() as i32
    }

    pub fn coordinate_system(from: &Self) -> (Self, Self) {
        let v1 = {
            if abs_t(from.x) > abs_t(from.y) {
                Vector3::new(-from.z, 0, from.x) /
                    ((from.x * from.x + from.z * from.z) as f32).sqrt() as i32
            } else {
                Vector3::new(0, from.z, -from.y) /
                    ((from.y * from.y + from.z * from.z) as f32).sqrt() as i32
            }
        };
        (v1, Vector3::cross(from, &v1))
    }
}

impl<T: NumberField> Vector3<T> {
    pub fn new(x: T, y: T, z: T) -> Self {
        Vector3 { x, y, z }
    }

    pub fn abs(v: &Self) -> Self {
        Vector3 {
            x: abs_t(v.x),
            y: abs_t(v.y),
            z: abs_t(v.z),
        }
    }

    pub fn cross(v1: &Self, v2: &Self) -> Self {
        let v1x = v1.x;
        let v1y = v1.y;
        let v1z = v1.z;
        let v2x = v2.x;
        let v2y = v2.y;
        let v2z = v2.z;
        Vector3 {
            x: (v1y * v2z) - (v1z * v2y),
            y: (v1z * v2x) - (v1x * v2z),
            z: (v1x * v2y) - (v1y * v2x),
        }
    }

    pub fn dot(v1: &Self, v2: &Self) -> T {
        v1.x * v2.x + v1.y * v2.y + v1.z * v2.z
    }

    pub fn abs_dot(v1: &Self, v2: &Self) -> T {
        abs_t(Vector3::dot(v1, v2))
    }

    pub fn min_component(v: &Self) -> T {
        min_t(v.x, min_t(v.y, v.z))
    }

    pub fn max_component(v: &Self) -> T {
        max_t(v.x, max_t(v.y, v.z))
    }

    pub fn max_dimensions(v: &Self) -> usize {
        if v.x > v.y {
            if v.x > v.z {
                0
            } else {
                2
            }
        } else {
            if v.y > v.z {
                1
            } else {
                2
            }
        }
    }

    pub fn min(v1: &Self, v2: &Self) -> Self {
        Vector3 {
            x: min_t(v1.x, v2.x),
            y: min_t(v1.y, v2.y),
            z: min_t(v1.z, v2.z),
        }
    }

    pub fn max(v1: &Self, v2: &Self) -> Self {
        Vector3 {
            x: max_t(v1.x, v2.x),
            y: max_t(v1.y, v2.y),
            z: max_t(v1.z, v2.z),
        }
    }

    pub fn permute(v: &Self, x: usize, y: usize, z: usize) -> Self {
        Vector3 {
            x: v[x],
            y: v[y],
            z: v[z],
        }
    }
}

impl<T: NumberField> Sub<Vector3<T>> for Vector3<T> {
    type Output = Vector3<T>;

    fn sub(self, rhs: Vector3<T>) -> Self::Output {
        Vector3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: NumberField> SubAssign<Vector3<T>> for Vector3<T> {
    fn sub_assign(&mut self, rhs: Vector3<T>) {
        self.x = self.x - rhs.x;
        self.y = self.y - rhs.y;
        self.z = self.z - rhs.z;
    }
}

impl<T: NumberField> Add<Vector3<T>> for Vector3<T> {
    type Output = Vector3<T>;

    fn add(self, rhs: Vector3<T>) -> Self::Output {
        Vector3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: NumberField> AddAssign<Vector3<T>> for Vector3<T> {
    fn add_assign(&mut self, rhs: Vector3<T>) {
        self.x = self.x + rhs.x;
        self.y = self.y + rhs.y;
        self.z = self.z + rhs.z;
    }
}

impl<T: NumberField> Neg for Vector3<T> {
    type Output = Vector3<T>;

    fn neg(self) -> Self::Output {
        Vector3 {
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }
}

impl Div<i32> for Vector3<i32> {
    type Output = Vector3<i32>;

    fn div(self, rhs: i32) -> Self::Output {
        let recip = 1.0 / (rhs as f32);
        Vector3 {
            x: ((self.x as f32) * recip) as i32,
            y: (self.y as f32 * recip) as i32,
            z: (self.z as f32 * recip) as i32,
        }
    }
}

impl Div<f32> for Vector3<f32> {
    type Output = Vector3<f32>;

    fn div(self, rhs: f32) -> Self::Output {
        let recip = 1.0 / rhs;
        Vector3 {
            x: self.x * recip,
            y: self.y * recip,
            z: self.z * recip,
        }
    }
}

impl DivAssign<i32> for Vector3<i32> {
    fn div_assign(&mut self, rhs: i32) {
        self.x = (self.x as f32 / rhs as f32) as i32;
        self.y = (self.y as f32 / rhs as f32) as i32;
        self.z = (self.z as f32 / rhs as f32) as i32;
    }
}

impl<T: NumberField> Mul<T> for Vector3<T> {
    type Output = Vector3<T>;

    fn mul(self, rhs: T) -> Self::Output {
        Vector3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl Mul<Vector3<i32>> for i32 {
    type Output = Vector3<i32>;

    fn mul(self, rhs: Vector3<i32>) -> Self::Output {
        rhs * self
    }
}

impl Mul<Vector3<f32>> for f32 {
    type Output = Vector3<f32>;

    fn mul(self, rhs: Vector3<f32>) -> Self::Output {
        rhs * self
    }
}

impl<T: NumberField> MulAssign<T> for Vector3<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.x = self.x * rhs;
        self.y = self.y * rhs;
        self.z = self.z * rhs;
    }
}

impl<T> Index<usize> for Vector3<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("index provided non-existent"),
        }
    }
}

impl<T> IndexMut<usize> for Vector3<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("index provided non-existent"),
        }
    }
}

fn abs_t<T: NumberField>(x: T) -> T {
    if x < T::default() {
        -x
    } else {
        x
    }
}

fn min_t<T: NumberField>(x: T, y: T) -> T {
    if x < y {
        x
    } else {
        y
    }
}

fn max_t<T: NumberField>(x: T, y: T) -> T {
    if x > y {
        x
    } else {
        y
    }
}

#[cfg(test)]
mod test_vector_ops {
    use super::{Vector2, Vector3};

    #[test]
    fn indexing() {
        let vec2 = Vector2 { x: 2, y: 3 };
        assert_eq!(vec2[0], 2);
        assert_eq!(vec2[1], 3);

        let vec3 = Vector3 { x: 4, y: 5, z: 6 };
        assert_eq!(vec3[0], 4);
        assert_eq!(vec3[1], 5);
        assert_eq!(vec3[2], 6);
    }

    #[test]
    fn creation() {
        let vec2 = Vector2::new(1, 2);
        assert_eq!(vec2.x, 1);
        assert_eq!(vec2.y, 2);

        let vec3 = Vector3::new(1, 2, 3);
        assert_eq!(vec3.x, 1);
        assert_eq!(vec3.y, 2);
        assert_eq!(vec3.z, 3);
    }

    #[test]
    fn addition_op() {
        let vec1 = Vector2::new(0, 2);
        let vec2 = Vector2::new(1, 3);
        let res1 = vec1 + vec2;
        assert_eq!(res1.x, 1);
        assert_eq!(res1.y, 5);

        let vec3 = Vector3::new(0, 2, 0);
        let vec4 = Vector3::new(1, 3, 1);
        let res2 = vec3 + vec4;
        assert_eq!(res2.x, 1);
        assert_eq!(res2.y, 5);
        assert_eq!(res2.z, 1);
    }

    #[test]
    fn addition_assign_op() {
        let mut vec1 = Vector2::new(0, 2);
        let vec2 = Vector2::new(1, 3);

        vec1 += vec2;
        assert_eq!(vec1.x, 1);
        assert_eq!(vec1.y, 5);

        let mut vec3 = Vector3::new(0, 2, 0);
        let vec4 = Vector3::new(1, 3, 1);

        vec3 += vec4;
        assert_eq!(vec3.x, 1);
        assert_eq!(vec3.y, 5);
        assert_eq!(vec3.z, 1);
    }

    #[test]
    fn subtraction_op() {
        let vec1 = Vector2::new(0, 2);
        let vec2 = Vector2::new(1, 3);
        let res1 = vec1 - vec2;
        assert_eq!(res1.x, -1);
        assert_eq!(res1.y, -1);

        let vec3 = Vector3::new(0, 2, 0);
        let vec4 = Vector3::new(1, 3, 1);
        let res2 = vec3 - vec4;
        assert_eq!(res2.x, -1);
        assert_eq!(res2.y, -1);
        assert_eq!(res2.z, -1);
    }

    #[test]
    fn subtraction_assign_op() {
        let mut vec1 = Vector2::new(0, 2);
        let vec2 = Vector2::new(1, 3);

        vec1 -= vec2;
        assert_eq!(vec1.x, -1);
        assert_eq!(vec1.y, -1);

        let mut vec3 = Vector3::new(0, 2, 0);
        let vec4 = Vector3::new(1, 3, 1);

        vec3 -= vec4;
        assert_eq!(vec3.x, -1);
        assert_eq!(vec3.y, -1);
        assert_eq!(vec3.z, -1);
    }

    #[test]
    fn scalar_mul() {
        let vec1 = Vector2::new(1, 1);
        let res = vec1 * 5;
        assert_eq!(res.x, 5);
        assert_eq!(res.y, 5);

        let vec2 = Vector3::new(3, 3, 3);
        let res2 = vec2 * 5;
        assert_eq!(res2.x, 15);
        assert_eq!(res2.y, 15);
        assert_eq!(res2.z, 15);

        // commutative test
        let vec3 = Vector3::new(3, 3, 3);
        let res3 = 5 * vec3;
        assert_eq!(res3.x, 15);
        assert_eq!(res3.y, 15);
        assert_eq!(res3.z, 15);

        let vec4 = Vector2::new(2, 2);
        let res4 = 4 * vec4;
        assert_eq!(res4.x, 8);
        assert_eq!(res4.y, 8);
    }

    #[test]
    fn scalar_mul_assign() {
        let mut vec1 = Vector2::new(1, 1);

        vec1 *= 5;
        assert_eq!(vec1.x, 5);
        assert_eq!(vec1.y, 5);

        let mut vec2 = Vector3::new(3, 3, 3);

        vec2 *= 5;
        assert_eq!(vec2.x, 15);
        assert_eq!(vec2.y, 15);
        assert_eq!(vec2.z, 15);
    }

    #[test]
    fn scalar_div() {
        let vec1 = Vector2::new(25, 25);
        let res = vec1 / 5;
        assert_eq!(res.x, 5);
        assert_eq!(res.y, 5);

        let vec2 = Vector3::new(30, 30, 30);
        let res2 = vec2 / 2;
        assert_eq!(res2.x, 15);
        assert_eq!(res2.y, 15);
        assert_eq!(res2.z, 15);
    }

    #[test]
    fn scalar_div_assign() {
        let mut vec1 = Vector2::new(25, 25);
        vec1 /= 5;
        assert_eq!(vec1.x, 5);
        assert_eq!(vec1.y, 5);

        let mut vec2 = Vector3::new(30, 30, 30);
        vec2 /= 2;
        assert_eq!(vec2.x, 15);
        assert_eq!(vec2.y, 15);
        assert_eq!(vec2.z, 15);
    }

    #[test]
    fn negation() {
        let vec1 = Vector2::new(3, 3);
        let res1 = -vec1;
        assert_eq!(res1.x, -3);
        assert_eq!(res1.y, -3);

        let vec2 = Vector3::new(1, 1, 1);
        let res2 = -vec2;
        assert_eq!(res2.x, -1);
        assert_eq!(res2.y, -1);
        assert_eq!(res2.z, -1);
    }

    #[test]
    fn abs() {
        let vec = Vector2::new(-2, -1);
        let vec = Vector2::abs(&vec);
        assert_eq!(vec.x, 2);
        assert_eq!(vec.y, 1);

        let vec = Vector3::new(-2, -1, -5);
        let vec = Vector3::abs(&vec);
        assert_eq!(vec.x, 2);
        assert_eq!(vec.y, 1);
        assert_eq!(vec.z, 5);
    }

    #[test]
    fn abs_dot() {
        let vec1 = Vector3::new(1, 0, 0);
        let vec2 = Vector3::new(-2, 1, 0);
        let res = Vector3::abs_dot(&vec1, &vec2);
        assert_eq!(2, res);

        let vec1 = Vector2::new(1, 0);
        let vec2 = Vector2::new(-2, 1);
        let res = Vector2::abs_dot(&vec1, &vec2);
        assert_eq!(2, res);
    }

    #[test]
    fn cross_product() {
        let vec1 = Vector3::new(1, 0, 0);
        let vec2 = Vector3::new(0, 1, 0);
        let perp_vector = Vector3::cross(&vec1, &vec2);
        assert_eq!(perp_vector, Vector3::new(0, 0, 1));
    }

    #[test]
    fn length_squared() {
        let vec = Vector2::new(1, 2);
        assert_eq!(5, vec.length_squared());

        let vec = Vector3::new(1, 2, 1);
        assert_eq!(6, vec.length_squared());
    }

    #[test]
    fn length() {
        let vec = Vector2::new(4, 3);
        assert_eq!(5.0, vec.length());

        let vec = Vector3::new(0, 4, 3);
        assert_eq!(5.0, vec.length());
    }
}
