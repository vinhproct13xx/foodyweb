from django.contrib import admin

# Register your models here.
from .models import Restaurants
from .models import CommentLikes
from .models import CommentPictures
from .models import Comments
from .models import Customers

admin.site.register(Restaurants)
admin.site.register(CommentLikes)
admin.site.register(CommentPictures)
admin.site.register(Comments)
admin.site.register(Customers)
